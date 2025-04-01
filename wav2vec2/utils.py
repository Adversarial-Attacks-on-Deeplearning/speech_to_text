import torch
import torchaudio


def preprocess_audio(file_path):
    """
    Preprocess the audio file by loading, resampling to 16kHz, converting to mono, 
    and normalizing the waveform to the range [-1, 1].

    Args:
        file_path (str): Path to the audio file.

    Returns:
        torch.Tensor: Preprocessed waveform.
    """
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0)
    waveform = waveform / waveform.abs().max()
    return waveform


def compute_transcription(input_data, model, processor, device):
    """
    Compute the transcription for a given audio file or waveform using the specified model and processor.

    Args:
        input_data (str or torch.Tensor): Path to the audio file or preprocessed waveform.
        model (torch.nn.Module): Pre-trained Wav2Vec2ForCTC model.
        processor (Wav2Vec2Processor): Processor to handle audio and text.
        device (str): Device to perform computation on ('cpu' or 'cuda').

    Returns:
        str: Transcription of the input audio or waveform.
    """
    if isinstance(input_data, str):  # If input_data is a file path
        waveform = preprocess_audio(input_data)
    elif isinstance(input_data, torch.Tensor):  # If input_data is already a waveform
        waveform = input_data
    else:
        raise ValueError("input_data must be a file path (str) or a waveform (torch.Tensor)")

    inputs = processor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # ensure inputs are on the specified device

    with torch.no_grad():
        logits = model(inputs["input_values"]).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription
