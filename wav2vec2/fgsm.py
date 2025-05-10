import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import jiwer
import numpy as np

def fgsm_attack(audio_tensors, target_transcription, model, processor, epsilon=0.3, sampling_rate=16000, device="cuda"):
    """
    Perform FGSM attack on Wav2Vec2 model for a batch of audio tensors.

    Args:
        audio_tensors (torch.Tensor): Batch of input audio waveforms (shape: [batch_size, max_length]).
        target_transcription (str): Desired target transcription.
        model (Wav2Vec2ForCTC): Pre-trained Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor.
        epsilon (float): Perturbation magnitude for FGSM.
        sampling_rate (int): Audio sampling rate.
        device (str): Device to run the model on.

    Returns:
        torch.Tensor: Adversarial waveforms (shape: [batch_size, max_length]).
    """
    # Move input to device
    input_values = audio_tensors.to(device)
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)  # Ensure batch dimension

    # Tokenize target transcription and move to device
    labels = processor.tokenizer(target_transcription, return_tensors="pt").input_ids.to(device)
    labels = labels.repeat(input_values.shape[0], 1)  # Repeat for batch

    # Enable gradient tracking
    input_values.requires_grad_(True)

    # Compute CTC loss
    output = model(input_values, labels=labels)
    loss = output.loss
    loss.backward()

    # Compute perturbation (targeted attack: minimize loss w.r.t. target)
    grad = input_values.grad
    perturbation = -epsilon * torch.sign(grad)

    # Create adversarial input
    adversarial_input_values = input_values.detach() + perturbation

    return adversarial_input_values.detach().cpu()
