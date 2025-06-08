import torch

def transcribe_audio(audio_array, sampling_rate, processor, model):
    if sampling_rate != 16000:
        raise ValueError(f"Expected 16kHz audio but got {sampling_rate}Hz")

    # Pass sampling_rate to the processor
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(model.device)

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription




import numpy as np
import IPython.display as ipd
# Function to calculate SNR
def calculate_snr(original, adversarial):
    # Ensure inputs are numpy arrays
    original = np.array(original)
    adversarial = np.array(adversarial)

    # Calculate noise (difference between adversarial and original)
    noise = adversarial - original

    # Calculate power of signal and noise
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)

    # Avoid division by zero
    if noise_power == 0:
        return float('inf')

    # Calculate SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr







from torch import nn
import torch
import torchaudio
def preprocess_audio(audio_array, sample_rate=16000,batch_dimension=True):
    # Ensure float32
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32) if isinstance(audio_array, np.ndarray) else audio_array

    # Resample if necessary (Wav2Vec2 expects 16 kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_tensor = resampler(audio_tensor)

    # Normalize (zero mean, unit variance)
    audio_tensor = (audio_tensor - audio_tensor.mean()) / (audio_tensor.std() + 1e-9)

    if batch_dimension:
        audio_tensor = audio_tensor.unsqueeze(0)

    return audio_tensor



# Custom collation function
def custom_collate_fn(batch):
    """
    Collate function to pad audio arrays to the same length within a batch.

    Args:
        batch: List of (audio_array, ground_truth, idx) tuples.

    Returns:
        Tuple of (padded_audio_tensor, ground_truth_list, indices_tensor).
    """
    audio_arrays, ground_truths, indices = zip(*batch)

    # Ensure audio_arrays are 1D NumPy arrays and convert to 1D PyTorch tensors
    audio_tensors = []
    for arr in audio_arrays:
        # Convert to NumPy if not already (handles potential tensors)
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        # Ensure 1D
        arr = np.ravel(arr)  # Flatten to 1D
        tensor = torch.as_tensor(arr, dtype=torch.float32)
        audio_tensors.append(tensor)

    # Pad to the longest in the batch
    padded_audio = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True, padding_value=0.0)

    # Convert ground truths to list (no padding needed)
    ground_truths = list(ground_truths)

    # Convert indices to tensor
    indices = torch.tensor(indices, dtype=torch.long)

    return padded_audio, ground_truths, indices


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor, sampling_rate=16000):
        self.dataset = dataset
        self.processor = processor
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        audio_array = example["audio"]["array"]
        ground_truth = example["text"]
        audio_array = preprocess_audio(audio_array)  # Assumed function
        return audio_array, ground_truth, idx
    



import numpy as np

def compute_segmental_snr(clean, processed, frame_len_ms=25, sample_rate=16000,
                         silence_threshold_db=-20, snr_clip_range=(-10, 35)):
    """
    Compute segmental Signal-to-Noise Ratio between clean and processed signals.

    Args:
        clean: Clean reference signal (numpy array or torch tensor)
        processed: Processed/noisy signal (numpy array or torch tensor)
        frame_len_ms: Frame length in milliseconds (default: 25ms)
        sample_rate: Sample rate in Hz (default: 16kHz)
        silence_threshold_db: Threshold below which frames are considered silent (default: -20dB)
        snr_clip_range: Tuple of (min_snr, max_snr) for clipping (default: (-10, 35))

    Returns:
        Average segmental SNR in dB, or np.nan if no valid frames
    """
    # Handle PyTorch tensors - convert to numpy
    if hasattr(clean, 'cpu'):  # Check if it's a torch tensor
        clean = clean.detach().cpu().numpy()
    if hasattr(processed, 'cpu'):  # Check if it's a torch tensor
        processed = processed.detach().cpu().numpy()

    # Ensure inputs are numpy arrays with proper dtype
    clean = np.asarray(clean, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)

    # Handle batched inputs (squeeze if needed)
    if clean.ndim > 1:
        clean = clean.squeeze()
    if processed.ndim > 1:
        processed = processed.squeeze()

    # Truncate to minimum length
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]

    # Compute noise signal
    noise = processed - clean

    # Calculate frame length in samples
    frame_len_samples = int(frame_len_ms * sample_rate / 1000)

    # Split signals into frames
    clean_frames = split_into_frames(clean, frame_len_samples)
    noise_frames = split_into_frames(noise, frame_len_samples)

    snr_values = []

    for clean_frame, noise_frame in zip(clean_frames, noise_frames):
        # Compute frame energies with small epsilon to avoid division by zero
        clean_energy = np.sum(clean_frame**2) + 1e-12
        noise_energy = np.sum(noise_frame**2) + 1e-12

        # Convert clean energy to dB for silence detection
        clean_energy_db = 10 * np.log10(clean_energy)

        # Skip silent frames
        if clean_energy_db < silence_threshold_db:
            continue

        # Compute frame SNR in dB
        frame_snr = 10 * np.log10(clean_energy / noise_energy)

        # Clip SNR to specified range
        clipped_snr = np.clip(frame_snr, snr_clip_range[0], snr_clip_range[1])
        snr_values.append(clipped_snr)

    # Return mean SNR or NaN if no valid frames
    return np.mean(snr_values) if snr_values else np.nan


def split_into_frames(signal, frame_len_samples, hop_len_samples=None):
    """
    Split signal into overlapping or non-overlapping frames.

    Args:
        signal: Input signal (numpy array)
        frame_len_samples: Frame length in samples
        hop_len_samples: Hop length in samples (default: frame_len_samples for non-overlapping)

    Returns:
        List of frames (numpy arrays)
    """
    if hop_len_samples is None:
        hop_len_samples = frame_len_samples

    frames = []
    start = 0

    while start + frame_len_samples <= len(signal):
        frame = signal[start:start + frame_len_samples]
        frames.append(frame)
        start += hop_len_samples

    return frames



import os
import json
from pathlib import Path

import json
import numpy as np
from pathlib import Path

def load_epsilon_group(epsilon, alpha=None, root_dir='adversarial_dataset'):
    root = Path(root_dir)
    # Build the glob pattern
    if alpha is not None:
        pattern = f"eps_{epsilon}_alpha_{alpha}"
    else:
        pattern = f"eps_{epsilon}*"
    # Find matching directories
    matching_dirs = [d for d in root.glob(pattern) if d.is_dir()]
    samples = []
    if not matching_dirs:
        print(f"No directories matching pattern '{pattern}' in {root}")
        return samples
    for param_dir in matching_dirs:
        json_files = list(param_dir.glob("*.json"))
        print(f"Found {len(json_files)} samples in {param_dir}")
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    metadata = json.load(f)
                numpy_path = metadata['numpy_path']
                if not Path(numpy_path).exists():
                    print(f"Warning: {numpy_path} does not exist")
                    continue
                audio_array = np.load(numpy_path)
                samples.append({
                    'audio': audio_array,
                    'sr': 16000,
                    'ground_truth': metadata['ground_truth'],
                    'params': metadata['params']
                })
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")
    return samples





def get_original_results(samples, processor, model):
    from tqdm import tqdm
    import evaluate 
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")
    cer_list = []
    wer_list = []
    print("Computing original (undefended) results...")
    for sample in tqdm(samples, desc="Processing original"):
        try:
            original_transcription = transcribe_audio(sample['audio'], 16000, processor, model)
            ground_truth = sample['ground_truth']
            cer = cer_metric.compute(predictions=[original_transcription], references=[ground_truth])
            wer = wer_metric.compute(predictions=[original_transcription], references=[ground_truth])
            cer_list.append(cer)
            wer_list.append(wer)
        except Exception as e:
            print(f"Error processing original sample: {str(e)}")
            continue
    if cer_list and wer_list:
        return {
            'cer': np.mean(cer_list),
            'wer': np.mean(wer_list),
            'num_samples': len(cer_list)
        }
    return None