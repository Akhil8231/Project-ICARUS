import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.fft import fft, fftfreq
from scipy.special import erfc

PRIM_POLY_RS = 0x13
GF_M = 4; GF_Q = 1 << GF_M; GF_N = 15; GF_K = 11; RS_T = 2
GF_EXP = np.zeros(2 * GF_N, dtype=np.int16)
GF_LOG = np.full(GF_Q, -1, dtype=np.int16)

def _gf_init():
    x = 1
    for i in range(GF_N):
        GF_EXP[i] = x
        if GF_LOG[x] == -1: GF_LOG[x] = i
        x <<= 1
        if x & GF_Q: x ^= PRIM_POLY_RS
    for i in range(GF_N, 2 * GF_N): GF_EXP[i] = GF_EXP[i - GF_N]
_gf_init()

def gf_mul(a, b):
    if a == 0 or b == 0: return 0
    return int(GF_EXP[(GF_LOG[a] + GF_LOG[b]) % GF_N])

def gf_div(a, b):
    if b == 0: raise ZeroDivisionError("GF division by zero")
    if a == 0: return 0
    return int(GF_EXP[(GF_LOG[a] - GF_LOG[b]) % GF_N])

def gf_pow(a, n):
    if a == 0: return 0
    return int(GF_EXP[(GF_LOG[a] * (n % GF_N)) % GF_N])

def poly_eval(poly, x):
    y = 0
    for c in poly: y = gf_mul(y, x) ^ c
    return y

def syndromes(cw, nsyn):
    return [poly_eval(cw, GF_EXP[i]) for i in range(1, nsyn + 1)]

def berlekamp_massey(S):
    C = [1] + [0] * len(S); B = [1] + [0] * len(S)
    L, m, b = 0, 1, 1
    for n in range(len(S)):
        d = S[n]
        for i in range(1, L + 1): d ^= gf_mul(C[i], S[n - i])
        if d != 0:
            T = C.copy()
            coef = gf_div(d, b)
            for i in range(m, len(C)): C[i] ^= gf_mul(coef, B[i - m])
            if 2 * L <= n: L, B, b, m = n + 1 - L, T, d, 1
            else: m += 1
        else: m += 1
    while len(C) > 0 and C[-1] == 0: C.pop()
    return C, L

def chien_search(locator):
    roots = []
    for i in range(GF_N):
        if poly_eval(locator, GF_EXP[(GF_N - i) % GF_N]) == 0: roots.append(i)
    return roots

def forney_magnitudes(S, locator, err_pos_pows):
    syndrome_poly = [0] + S
    omega_full = [0] * (len(syndrome_poly) + len(locator))
    for i, c1 in enumerate(syndrome_poly):
        for j, c2 in enumerate(locator): omega_full[i+j] ^= gf_mul(c1, c2)
    omega_poly = omega_full[:2*RS_T+1]
    deriv_coeffs = locator[1::2]
    mags = []
    for pos in err_pos_pows:
        xinvi = GF_EXP[(GF_N - pos) % GF_N]
        num = poly_eval(omega_poly, xinvi)
        den = poly_eval(deriv_coeffs, gf_pow(xinvi, 2))
        if den == 0: mags.append(0); continue
        mags.append(gf_div(num, den))
    return mags

def bits_to_nibbles_msb(bits):
    return [int("".join(map(str, bits[i:i+4])), 2) for i in range(0, len(bits), 4)]

def bits_to_nibbles_lsb(bits):
    return [int("".join(map(str, bits[i:i+4][::-1])), 2) for i in range(0, len(bits), 4)]

def nibbles_to_bits_msb(nibs):
    return np.array([int(b) for nib in nibs for b in f'{nib:04b}'], dtype=np.int8)

def _rs1511_decode_blocks(sym_list):
    data_out, valid_blocks, total_blocks = [], 0, 0
    for i in range(0, len(sym_list) - (len(sym_list) % GF_N), GF_N):
        total_blocks += 1
        cw = sym_list[i:i+GF_N]
        S = syndromes(cw, 2 * RS_T)
        if not np.any(S):
            data_out.extend(cw[:GF_K]); valid_blocks += 1; continue
        locator, L = berlekamp_massey(S)
        if L == 0 or L > RS_T:
            data_out.extend(cw[:GF_K]); continue
        err_pos = chien_search(locator)
        if len(err_pos) != L:
            data_out.extend(cw[:GF_K]); continue
        mags = forney_magnitudes(S, locator, err_pos)
        cw_corr = cw[:]
        for idx, mag in zip([GF_N - 1 - p for p in err_pos], mags):
            if 0 <= idx < GF_N: cw_corr[idx] ^= mag
        if not np.any(syndromes(cw_corr, 2 * RS_T)):
            data_out.extend(cw_corr[:GF_K]); valid_blocks += 1
        else: data_out.extend(cw[:GF_K])
    return data_out, valid_blocks, total_blocks

def rs1511_decode(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    n_nibbles = len(bits) // 4
    if n_nibbles < GF_N: return np.zeros(0, dtype=np.int8)
    syms_msb = bits_to_nibbles_msb(bits[:n_nibbles * 4])
    data_msb, ok_msb, _ = _rs1511_decode_blocks(syms_msb)
    syms_lsb = bits_to_nibbles_lsb(bits[:n_nibbles * 4])
    data_lsb, ok_lsb, _ = _rs1511_decode_blocks(syms_lsb)
    chosen = data_lsb if ok_lsb > ok_msb else data_msb
    return nibbles_to_bits_msb(chosen)

# --- A.2: Viterbi Decoder ---
def viterbi_decode_soft(llrs):
    G = [0o133, 0o171]; K = 7
    n_states = 1 << (K - 1)
    n_bits = len(llrs) // 2
    next_state = np.zeros((n_states, 2), dtype=int)
    outputs = np.zeros((n_states, 2, 2), dtype=int)
    for state in range(n_states):
        for inp in [0, 1]:
            shift_reg = (state << 1) | inp
            out = [bin(shift_reg & g).count("1") % 2 for g in G]
            outputs[state, inp] = out
            next_state[state, inp] = shift_reg & (n_states - 1)
    path_metrics = np.full(n_states, np.inf); path_metrics[0] = 0
    paths = np.full((n_states, n_bits), -1, dtype=int)
    for t in range(n_bits):
        new_metrics = np.full(n_states, np.inf)
        new_paths = np.full((n_states, n_bits), -1, dtype=int)
        rx_llr = llrs[2 * t:2 * t + 2]
        for state in range(n_states):
            if path_metrics[state] < np.inf:
                for inp in [0, 1]:
                    ns = next_state[state, inp]
                    exp_out = outputs[state, inp]
                    branch_metric = -np.sum(rx_llr * (1 - 2 * np.array(exp_out)))
                    metric = path_metrics[state] + branch_metric
                    if metric < new_metrics[ns]:
                        new_metrics[ns] = metric; new_paths[ns] = paths[state]; new_paths[ns, t] = inp
        path_metrics = new_metrics; paths = new_paths
    best_state = np.argmin(path_metrics)
    decoded = paths[best_state]
    return decoded[decoded >= 0]

def bpsk_llr(symbols, noise_var=1.0):
    llrs = 2 * np.real(symbols) / (noise_var + 1e-12)
    return np.clip(llrs, -50, 50)

# --- A.3: Doppler Correction ---
def correct_doppler(rx_signal, fs=1.0):
    squared_signal = rx_signal ** 2
    n = len(squared_signal)
    fft_result = fft(squared_signal, n * 4)
    freqs = fftfreq(n * 4, 1/fs)
    peak_index = np.argmax(np.abs(fft_result[1:])) + 1
    estimated_offset = freqs[peak_index] / 2.0
    time_vector = np.arange(n) / fs
    correction = np.exp(-1j * 2 * np.pi * estimated_offset * time_vector)
    return rx_signal * correction, estimated_offset

def process_phase1(sample_path):
    print(f"--- Processing Phase 1: {sample_path} ---")
    rx_signal = np.load(os.path.join(sample_path, 'rx.npy'))
    with open(os.path.join(sample_path, 'meta.json'), 'r') as f:
        metadata = json.load(f)
    sps = metadata['sps']
    ground_truth_bits = np.array(metadata['clean_bits'])
    pulse_shape = np.ones(sps) / np.linalg.norm(np.ones(sps))
    filtered_signal = np.convolve(rx_signal, pulse_shape, mode='full')
    offset = np.argmax(np.abs(filtered_signal[0:2*sps]))
    symbol_samples = filtered_signal[offset::sps]
    decoded_bits = (np.real(symbol_samples) > 0).astype(int)
    min_len = min(len(decoded_bits), len(ground_truth_bits))
    decoded_bits, ground_truth_bits = decoded_bits[:min_len], ground_truth_bits[:min_len]
    ber = np.sum(decoded_bits != ground_truth_bits) / len(ground_truth_bits)
    np.save(os.path.join(sample_path, 'decoded_bits.npy'), decoded_bits)
    return {'ber': ber, 'snr_db': metadata['snr_db'], 'coding': 'none', 'symbol_samples': symbol_samples}

def process_phase2(sample_path):
    print(f"--- Processing Phase 2: {sample_path} ---")
    rx_signal = np.load(os.path.join(sample_path, 'rx.npy'))
    with open(os.path.join(sample_path, 'meta.json'), 'r') as f:
        metadata = json.load(f)
    sps = metadata['sps']
    ground_truth_bits = np.array(metadata['clean_bits'])
    pulse_shape = np.ones(sps) / np.linalg.norm(np.ones(sps))
    filtered_signal = fftconvolve(rx_signal, pulse_shape, mode='full')
    energies = np.array([np.mean(np.abs(filtered_signal[i::sps])**2) for i in range(sps)])
    offset = np.argmax(energies)
    symbol_samples = filtered_signal[offset::sps][:len(ground_truth_bits)]
    decoded_bits = (np.real(symbol_samples) > 0).astype(int)
    ideal_symbols = 2 * ground_truth_bits - 1
    noise = symbol_samples - ideal_symbols
    signal_power = np.mean(np.abs(ideal_symbols)**2)
    noise_power = np.var(noise)
    estimated_snr_db = 10 * np.log10(signal_power / noise_power)
    min_len = min(len(decoded_bits), len(ground_truth_bits))
    decoded_bits, ground_truth_bits = decoded_bits[:min_len], ground_truth_bits[:min_len]
    ber = np.sum(decoded_bits != ground_truth_bits) / len(ground_truth_bits)
    np.save(os.path.join(sample_path, 'decoded_bits.npy'), decoded_bits)
    return {'ber': ber, 'snr_db': metadata['snr_db'], 'estimated_snr_db': estimated_snr_db, 'coding': 'none', 'symbol_samples': symbol_samples}

def process_advanced(sample_path):
    """Processes Phase 3 (Coding) and 4 (Doppler) using the advanced receiver."""
    print(f"--- Processing Advanced: {sample_path} ---")
    rx_signal = np.load(os.path.join(sample_path, 'rx.npy'))
    with open(os.path.join(sample_path, 'meta.json'), 'r') as f:
        metadata = json.load(f)
    sps = metadata['sps']; ground_truth_bits = np.array(metadata['clean_bits'])
    coding = metadata.get('coding', 'none')
    num_symbols = metadata.get('num_symbols')
    if num_symbols is None:
        num_info_bits = len(ground_truth_bits)
        if coding == 'conv': num_symbols = num_info_bits * 2
        elif coding == 'rs': num_symbols = int(np.ceil(num_info_bits / (GF_K * GF_M))) * GF_N
        else: num_symbols = num_info_bits
    is_doppler_phase = 'phase4_doppler' in sample_path
    original_signal_for_plot = rx_signal.copy() if is_doppler_phase else None
    if is_doppler_phase: rx_signal, _ = correct_doppler(rx_signal)
    pulse_shape = np.ones(sps) / np.linalg.norm(np.ones(sps))
    filtered_signal = fftconvolve(rx_signal, pulse_shape, mode='full')
    energies = np.array([np.mean(np.abs(filtered_signal[i::sps])**2) for i in range(sps)])
    offset = np.argmax(energies)
    symbol_samples = filtered_signal[offset::sps][:num_symbols]
    avg_phase_vector = np.mean(symbol_samples**2)
    estimated_phase_offset = np.angle(avg_phase_vector) / 2.0
    phase_correction = np.exp(-1j * estimated_phase_offset)
    corrected_samples = symbol_samples * phase_correction
    if coding == 'conv':
        noise_var = np.var(np.imag(corrected_samples))
        llrs = bpsk_llr(corrected_samples, noise_var)
        decoded_bits = viterbi_decode_soft(llrs)
    else:
        hard_bits = (np.real(corrected_samples) > 0).astype(int)
        if coding == 'rs': decoded_bits = rs1511_decode(hard_bits)
        else: decoded_bits = hard_bits
    min_len = min(len(decoded_bits), len(ground_truth_bits))
    decoded_bits, ground_truth_bits = decoded_bits[:min_len], ground_truth_bits[:min_len]
    ber = np.sum(decoded_bits != ground_truth_bits) / len(ground_truth_bits) if len(ground_truth_bits) > 0 else 0
    fer = None
    if coding == 'rs' and len(ground_truth_bits) > 0:
        frame_len = GF_K * GF_M; num_frames = len(ground_truth_bits) // frame_len
        frame_errors = sum(1 for i in range(num_frames) if np.any(decoded_bits[i*frame_len:(i+1)*frame_len] != ground_truth_bits[i*frame_len:(i+1)*frame_len]))
        fer = frame_errors / num_frames if num_frames > 0 else 0
    np.save(os.path.join(sample_path, 'decoded_bits.npy'), decoded_bits)
    result = {'ber': ber, 'fer': fer, 'snr_db': metadata['snr_db'], 'coding': coding, 'symbol_samples': symbol_samples}
    if is_doppler_phase:
        result.update({'doppler_corrected_signal': rx_signal, 'doppler_original_signal': original_signal_for_plot})
    return result

def plot_all_results(results):
    # Plot 1: BER vs SNR
    plt.figure(figsize=(12, 8))
    results_uncoded = [r for r in results if r['coding'] == 'none']
    results_conv = [r for r in results if r['coding'] == 'conv']
    results_rs = [r for r in results if r['coding'] == 'rs']
    plt.scatter([r['snr_db'] for r in results_uncoded], [r['ber'] for r in results_uncoded], marker='o', label='Uncoded BPSK (BER)')
    plt.scatter([r['snr_db'] for r in results_conv], [r['ber'] for r in results_conv], marker='x', s=100, label='Conv. Coded (BER)')
    plt.scatter([r['snr_db'] for r in results_rs], [r['ber'] for r in results_rs], marker='s', label='RS Coded (BER)')
    all_snrs = [r['snr_db'] for r in results];
    snr_range = np.linspace(min(all_snrs) - 1, max(all_snrs) + 1, 100)
    snr_linear = 10**(snr_range / 10.0)
    plt.plot(snr_range, 0.5 * erfc(np.sqrt(snr_linear)), 'k--', label='Theoretical Uncoded BPSK')
    plt.yscale('log'); plt.ylim(1e-6, 1.0); plt.xlim(min(all_snrs)-1, max(all_snrs)+1)
    plt.title('Performance vs. SNR'); plt.xlabel('SNR (dB)'); plt.ylabel('Error Rate (BER/FER)')
    plt.grid(True, which='both', linestyle=':'); plt.legend(); plt.savefig('ber_fer_vs_snr.png')
    print("Saved BER/FER vs SNR plot to ber_fer_vs_snr.png")

    # Plot 2: Constellations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    uncoded_results = [r for r in results if 'symbol_samples' in r and r['coding'] == 'none']
    if uncoded_results:
        low_snr_sample = min(uncoded_results, key=lambda x: x['snr_db'])
        high_snr_sample = max(uncoded_results, key=lambda x: x['snr_db'])
        for ax, sample, name in [(ax1, low_snr_sample, "Low"), (ax2, high_snr_sample, "High")]:
            ax.scatter(np.real(sample['symbol_samples']), np.imag(sample['symbol_samples']), alpha=0.5)
            ax.set_title(f'Constellation at {name} SNR ({sample["snr_db"]} dB)'); ax.set_xlabel('In-Phase (I)'); ax.set_ylabel('Quadrature (Q)')
            ax.grid(True); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal', adjustable='box')
    plt.tight_layout(); plt.savefig('constellation_diagrams.png')
    print("Saved constellation diagrams to constellation_diagrams.png")

    # Plot 3: Doppler Correction
    doppler_sample = next((r for r in results if 'doppler_original_signal' in r), None)
    if doppler_sample:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, signal, name in [(ax1, doppler_sample['doppler_original_signal'], "Before"), (ax2, doppler_sample['doppler_corrected_signal'], "After")]:
            n = len(signal)
            fft_abs = np.abs(fft(signal) / n)
            freqs = fftfreq(n, 1.0)
            ax.plot(np.fft.fftshift(freqs), np.fft.fftshift(20*np.log10(fft_abs)))
            ax.set_title(f'Spectrum {name} Doppler Correction'); ax.set_xlabel('Normalized Frequency'); ax.set_ylabel('Magnitude (dB)')
            ax.grid(True)
        plt.tight_layout(); plt.savefig('doppler_compensation.png')
        print("Saved Doppler compensation plot to doppler_compensation.png")
    
    plt.show()


if __name__ == "__main__":
    base_path = 'cubesat_dataset'
    all_results = []
    if not os.path.isdir(base_path):
        print(f"Error: Dataset directory '{base_path}' not found.")
    else:
        # Phase 1
        phase1_path = os.path.join(base_path, 'phase1_timing')
        if os.path.isdir(phase1_path):
            for dirpath, _, filenames in os.walk(phase1_path):
                if 'rx.npy' in filenames: all_results.append(process_phase1(dirpath))
        # Phase 2
        phase2_path = os.path.join(base_path, 'phase2_snr')
        if os.path.isdir(phase2_path):
            for dirpath, _, filenames in os.walk(phase2_path):
                if 'rx.npy' in filenames: all_results.append(process_phase2(dirpath))
        # Phase 3
        phase3_path = os.path.join(base_path, 'phase3_coding')
        if os.path.isdir(phase3_path):
            for dirpath, _, filenames in os.walk(phase3_path):
                if 'rx.npy' in filenames: all_results.append(process_advanced(dirpath))
        # Phase 4
        phase4_path = os.path.join(base_path, 'phase4_doppler')
        if os.path.isdir(phase4_path):
            for dirpath, _, filenames in os.walk(phase4_path):
                if 'rx.npy' in filenames: all_results.append(process_advanced(dirpath))
        
        # --- Print Summary & Plot ---
        print("\n" + "="*60 + "\n--- Performance Metrics Summary ---\n" + "="*60)
        if all_results:
            for res in sorted(all_results, key=lambda r: r.get('path', '')):
                ber, snr, coding = res['ber'], res['snr_db'], res['coding']
                print(f"SNR: {snr:>2}dB | Coding: {coding:<11} | BER: {ber:.2e}")
            plot_all_results(all_results)
        else:
            print("No samples were processed.")
