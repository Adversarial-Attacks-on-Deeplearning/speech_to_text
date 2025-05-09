import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import jiwer
import numpy as np

def fgsm_attack(audio_array, ground_truth, target_transcription, model, processor, epsilon=0.3, sampling_rate=16000):
    """
    Perform FGSM attack on Wav2Vec2 model to generate adversarial audio.

    Args:
        audio_array (np.ndarray): Input audio waveform (1D NumPy array).
        ground_truth (str): Ground truth transcription.
        target_transcription (str): Desired target transcription for the attack.
        model (Wav2Vec2ForCTC): Pre-trained Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor for audio and text processing.
        epsilon (float): Perturbation magnitude for FGSM. Default is 0.3.
        sampling_rate (int): Audio sampling rate (default: 16000 Hz).
        device (str): Device to run the model on (default: "cuda").

    Returns:
        tuple: (adversarial_waveform, ground_truth_wer, target_wer, adversarial_transcription)
            - adversarial_waveform (np.ndarray): Perturbed audio waveform.
            - ground_truth_wer (float): WER between ground truth and adversarial transcription.
            - target_wer (float): WER between target transcription and adversarial transcription.
            - adversarial_transcription (str): Transcription of the adversarial audio.
    """
    # Step 1: Preprocess the audio
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(model.device)  # Shape: [1, audio_length]

    # Step 2: Tokenize the target transcription
    labels = processor.tokenizer(target_transcription, return_tensors="pt").input_ids

    # Step 3 & 4: Compute CTC loss with gradient tracking
    input_values.requires_grad_(True)
    output = model(input_values, labels=labels)
    loss = output.loss
    loss.backward()

    # Step 5: Compute the gradient
    grad = input_values.grad  # Gradient of loss w.r.t. input_values

    # Step 6: Generate perturbation (targeted attack: minimize loss w.r.t. target)
    perturbation = -epsilon * torch.sign(grad)

    # Step 7: Create adversarial input
    adversarial_input_values = input_values.detach() + perturbation


    # Convert adversarial input to NumPy array for return
    adversarial_waveform = adversarial_input_values.squeeze().cpu().numpy()

    return adversarial_waveform