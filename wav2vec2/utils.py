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


# Custom dataset (unchanged)
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
        ground_truth = example["true_text"]
        audio_array = preprocess_audio(audio_array)  # Assumed function
        return audio_array, ground_truth, idx
    


