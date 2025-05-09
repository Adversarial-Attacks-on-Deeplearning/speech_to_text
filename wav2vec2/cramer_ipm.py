import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import IPython.display as ipd
import numpy as np



# Cramér-IPM computation
def compute_cramer_ipm(original_input, adversarial_input):
    """
    Compute a simplified Cramér-IPM between original and adversarial inputs.
    Uses mean squared error as a placeholder for distribution difference.

    Args:
        original_input (torch.Tensor): Original audio tensor.
        adversarial_input (torch.Tensor): Adversarial audio tensor.

    Returns:
        torch.Tensor: Approximated Cramér-IPM value.
    """
    return torch.mean((original_input - adversarial_input) ** 2)

def cramer_ipm_attack(audio_array, ground_truth, target_transcription, model, processor,
                      epsilon=0.01, sampling_rate=16000, num_iterations=10, lambda_ipm=1):
    """
    Perform Cramér-IPM-based attack on Wav2Vec2 model to generate adversarial audio.

    Args:
        audio_array (np.ndarray): Input audio waveform (1D NumPy array).
        ground_truth (str): Ground truth transcription (unused in this function).
        target_transcription (str): Desired target transcription for the attack.
        model (Wav2Vec2ForCTC): Pre-trained Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor for audio and text processing.
        epsilon (float): Perturbation magnitude per iteration. Default is 0.01.
        sampling_rate (int): Audio sampling rate. Default is 16000 Hz.
        num_iterations (int): Number of iterations for the attack. Default is 10.
        lambda_ipm (float): Weight for Cramér-IPM term in the optimization. Default is 1.

    Returns:
        np.ndarray: Adversarial audio waveform.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Preprocess the audio
    inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to(device)
    input_values.requires_grad_(True)

    # Tokenize the target transcription
    labels = processor.tokenizer(target_transcription, return_tensors="pt").input_ids.to(device)

    # Initialize adversarial input
    adversarial_input_values = input_values.clone().detach().requires_grad_(True)

    # Iterative attack
    for i in range(num_iterations):
        # Compute CTC loss
        output = model(adversarial_input_values, labels=labels)
        ctc_loss = output.loss

        # Compute gradients for CTC loss
        model.zero_grad()
        ctc_loss.backward()
        grad_ctc = adversarial_input_values.grad.clone()

        # Reset gradients
        adversarial_input_values.grad.zero_()

        # Compute Cramér-IPM
        cramér_ipm = compute_cramer_ipm(input_values, adversarial_input_values)

        # Compute gradients for Cramér-IPM
        cramér_ipm.backward()
        grad_ipm = adversarial_input_values.grad.clone()

        # Combine gradients
        grad = grad_ctc - lambda_ipm * grad_ipm

        # Generate perturbation
        perturbation = -epsilon * torch.sign(grad)

        # Update adversarial input
        adversarial_input_values = adversarial_input_values + perturbation
        adversarial_input_values = torch.clamp(adversarial_input_values, min=-1, max=1)
        adversarial_input_values = adversarial_input_values.detach().requires_grad_(True)


    # Convert to NumPy array
    adversarial_waveform = adversarial_input_values.squeeze().detach().cpu().numpy()

    return adversarial_waveform


