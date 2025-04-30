import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer
import numpy as np

def pgd_attack(audio_array, ground_truth, target_transcription, model, processor, epsilon=0.3, alpha=0.01, num_iter=40, sampling_rate=16000, device="cuda"):
    """
    Perform PGD attack on Wav2Vec2 model to generate adversarial audio.

    Args:
        audio_array (np.ndarray): Input audio waveform (1D NumPy array).
        ground_truth (str): Ground truth transcription.
        target_transcription (str): Desired target transcription for the attack.
        model (Wav2Vec2ForCTC): Pre-trained Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor for audio and text processing.
        epsilon (float): Maximum perturbation magnitude. Default is 0.3.
        alpha (float): Step size for each iteration. Default is 0.01.
        num_iter (int): Number of iterations for the attack. Default is 40.
        sampling_rate (int): Audio sampling rate (default: 16000 Hz).
        device (str): Device to run the model on (default: "cuda").

    Returns:
        tuple: (adversarial_waveform, ground_truth_wer, target_wer, adversarial_transcription)
            - adversarial_waveform (np.ndarray): Perturbed audio waveform.
            - ground_truth_wer (float): WER between ground truth and adversarial transcription.
            - target_wer (float): WER between target transcription and adversarial transcription.
            - adversarial_transcription (str): Transcription of the adversarial audio.
    """
    # Preprocess the audio
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
    input_values = inputs.input_values  # Shape: [1, audio_length]

    # Tokenize the target transcription
    labels = processor.tokenizer(target_transcription, return_tensors="pt").input_ids

    # Initialize perturbation
    perturbation = torch.zeros_like(input_values)
    perturbation.requires_grad_(True)

    # Iterative PGD attack
    for i in range(num_iter):
        # Compute CTC loss
        output = model(input_values + perturbation, labels=labels)
        loss = output.loss
        loss.backward()

        # Get gradient of perturbation
        grad = perturbation.grad

        # Update perturbation: step in the direction to minimize loss (targeted attack)
        perturbation = perturbation - alpha * torch.sign(grad)

        # Project perturbation to be within [-epsilon, epsilon]
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        perturbation = perturbation.detach().requires_grad_(True)
        if i % 10 == 0 and i>0:
          print(f"Iteration {i+1}/{num_iter} ")

    # Create adversarial input
    adversarial_input_values = input_values + perturbation

    # Transcribe adversarial audio
    with torch.no_grad():
        logits = model(adversarial_input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        adversarial_transcription = processor.batch_decode(predicted_ids)[0]

    # Evaluate the attack
    ground_truth_wer = jiwer.wer(ground_truth, adversarial_transcription)
    target_wer = jiwer.wer(target_transcription, adversarial_transcription)

    # Convert adversarial input to NumPy array for return
    adversarial_waveform = adversarial_input_values.detach().squeeze().cpu().numpy()

    return adversarial_waveform, ground_truth_wer, target_wer, adversarial_transcription




