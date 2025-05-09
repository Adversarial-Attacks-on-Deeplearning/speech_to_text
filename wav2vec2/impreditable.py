import numpy as np
import librosa
from scipy.signal import find_peaks
from datasets import load_dataset

def freq_to_bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)

def threshold_in_quiet(f):
    f_khz = f / 1000
    if f_khz == 0:
        return 100  # High threshold for 0 Hz (inaudible)
    term1 = 3.64 * f_khz**-0.8
    term2 = -6.5 * np.exp(-0.6 * (f_khz - 3.3)**2)
    term3 = 0.001 * f_khz**4
    return term1 + term2 + term3

def compute_masking_threshold(x, fs=16000, N=2048, hop=512):
    # Compute STFT
    win = librosa.filters.get_window('hann', N, fftbins=True)
    stft = librosa.stft(x, n_fft=N, hop_length=hop, window=win)
    mag_sq = np.abs(stft)**2 / N
    p_x = 10 * np.log10(mag_sq + 1e-10)  # Shape: (freq, time) = (1025, num_frames)

    # Frequency and Bark scales
    freqs = np.fft.rfftfreq(N, d=1/fs)
    bark = np.array([freq_to_bark(f) for f in freqs])
    TIQ = np.array([threshold_in_quiet(f) for f in freqs])

    # Initialize output
    num_frames = p_x.shape[1]  # Number of time frames
    num_bins = p_x.shape[0]    # Number of frequency bins
    theta_x = np.zeros((num_frames, num_bins))

    # Compute masking threshold for each frame
    for t in range(num_frames):
        p = p_x[:, t]  # PSD for frame t, shape: (1025,)

        # Identify maskers
        peaks, _ = find_peaks(p)
        maskers = []
        for k in peaks:
            if p[k] > TIQ[k]:
                b_k = bark[k]
                neighborhood = np.abs(bark - b_k) < 0.5
                if p[k] == np.max(p[neighborhood]):
                    maskers.append(k)

        '# Compute global masking threshold'
        for k in range(num_bins):
            sum_mask = 0
            for k_m in maskers:
                delta_bark = np.abs(bark[k] - bark[k_m])
                if bark[k] < bark[k_m]:
                    s = 27
                else:
                    s = 24 - 0.2 * p[k_m]
                M_m_k = p[k_m] - s * delta_bark
                sum_mask += 10**(M_m_k / 10)
            theta_x[t, k] = 10 * np.log10(10**(TIQ[k] / 10) + sum_mask)

    return theta_x

# Extract audio arrays, sampling rates, and ground truths
audio_arrays = [example["audio"]["array"] for example in librispeech]
sampling_rates = [example["audio"]["sampling_rate"] for example in librispeech]
ground_truths = [example["true_text"].lower().strip() for example in librispeech]  # Adjusted 'true_text' to 'text'

# Test the function on the first sample
x = audio_arrays[0]
fs = sampling_rates[0]

# Compute masking threshold
theta_x = compute_masking_threshold(x, fs=fs, N=2048, hop=512)

# Print results
print("Audio sample length:", len(x), "samples")
print("Sampling rate:", fs, "Hz")
print("Ground truth transcription:", ground_truths[0])
print("Masking threshold shape:", theta_x.shape)