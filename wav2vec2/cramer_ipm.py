import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor



def compute_cramer_ipm(original_input, adversarial_input, n_samples=100):
  """
  Compute Cramér-IPM between original and adversarial inputs for a batch.
  """
  batch_size = original_input.shape[0]
  ipm_values = torch.zeros(batch_size, device=original_input.device)

  # Ensure adversarial_input requires grad
  adversarial_input = adversarial_input.clone().requires_grad_(True)

  for i in range(batch_size):
      # Get distribution samples for original and adversarial inputs
      orig_dist = original_input[i].view(-1)
      adv_dist = adversarial_input[i].view(-1)

      # Determine integration range
      min_val = min(orig_dist.min().item(), adv_dist.min().item())
      max_val = max(orig_dist.max().item(), adv_dist.max().item())

      # Sample points for integration
      sample_points = torch.linspace(min_val, max_val, n_samples, device=original_input.device)

      # Compute empirical CDFs directly at sample points
      orig_cdf_vals = compute_cdf_at_points(sample_points, orig_dist)
      adv_cdf_vals = compute_cdf_at_points(sample_points, adv_dist)

      # Compute L2 distance between CDFs (Cramér distance)
      # Use trapezoidal rule for numerical integration
      point_distances = (orig_cdf_vals - adv_cdf_vals) ** 2
      dx = (max_val - min_val) / (n_samples - 1)
      cramer_distance = torch.sum(point_distances) * dx

      ipm_values[i] = cramer_distance

  return ipm_values


def compute_cdf_at_points(points, samples):
    """
    Compute empirical CDF values directly at specified points.
    More efficient than sorting and interpolating.
    """
    n_samples = len(samples)
    cdf_values = torch.zeros_like(points)

    for i, point in enumerate(points):
        # Count how many samples are <= point
        count = torch.sum(samples <= point).float()
        cdf_values[i] = count / n_samples

    return cdf_values


def cramer_ipm_attack(audio_tensors, target_transcription, model, processor,
                       epsilon=0.01, sampling_rate=16000, num_iterations=10,
                       lambda_ipm=1.0, n_samples=100, device="cuda"):
    """
    Perform Cramér-IPM-based attack on speech model for a batch of audio tensors.
    Fixed version to ensure proper gradient handling and SNR calculation.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Move input to device
    input_values = audio_tensors.to(device)
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)  # Ensure batch dimension

    # Tokenize target transcription and move to device
    labels = processor.tokenizer(target_transcription, return_tensors="pt").input_ids.to(device)
    labels = labels.repeat(input_values.shape[0], 1)  # Repeat for batch

    # Initialize adversarial input
    adversarial_input_values = input_values.clone().detach().requires_grad_(True)

    # For logging
    snr_values = []
    loss_values = []

    # Iterative attack
    for i in range(num_iterations):
        # Create a copy with gradients enabled for this iteration
        adv_input = adversarial_input_values.clone().detach().requires_grad_(True)

        # Step 1: Compute CTC loss and its gradient
        model.zero_grad()
        outputs = model(adv_input, labels=labels)
        ctc_loss = outputs.loss
        ctc_loss.backward()
        grad_ctc = adv_input.grad.clone()

        # Step 2: Compute Cramér-IPM separately
        with torch.no_grad():
            cramer_ipm_val = compute_cramer_ipm(input_values, adv_input.detach(), n_samples)

        # Step 3: Approximate Cramér-IPM gradient using finite differences
        adv_input_ipm = adversarial_input_values.clone().detach().requires_grad_(True)
        perturb = torch.randn_like(adv_input_ipm) * 1e-5  # Small random perturbation
        adv_input_perturbed = adv_input_ipm + perturb
        cramer_ipm_val = compute_cramer_ipm(input_values, adv_input_ipm, n_samples)
        cramer_ipm_val_perturbed = compute_cramer_ipm(input_values, adv_input_perturbed, n_samples)
        grad_ipm = (cramer_ipm_val_perturbed - cramer_ipm_val).mean() / 1e-5 * perturb

        # Step 4: Combine gradients with lambda weighting
        combined_grad = grad_ctc - lambda_ipm * grad_ipm

        # Step 5: Generate perturbation using sign method
        perturbation = epsilon * torch.sign(combined_grad)

        # Step 6: Update adversarial input
        with torch.no_grad():
            adversarial_input_values = adversarial_input_values - perturbation
            adversarial_input_values = torch.clamp(adversarial_input_values, min=-1.0, max=1.0)

        # Step 7: Calculate SNR for monitoring
        with torch.no_grad():
            noise = adversarial_input_values - input_values
            signal_power = torch.mean(input_values ** 2, dim=1)
            noise_power = torch.mean(noise ** 2, dim=1)
            batch_snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
            avg_snr = batch_snr.mean().item()
            snr_values.append(avg_snr)
            loss_values.append(ctc_loss.item())

        # Print progress for debugging
        if i % 2 == 0:
            print(f"Iteration {i}/{num_iterations}, Loss: {ctc_loss.item():.4f}, SNR: {avg_snr:.2f} dB")

    return adversarial_input_values.detach().cpu(), {"snr_values": snr_values, "loss_values": loss_values}



def compute_segmental_snr(original, adversarial, frame_length=512, hop_length=256):
    """
    Compute segmental Signal-to-Noise Ratio (segSNR) between original and adversarial audio.
    This is a more accurate measure used in the paper.

    Args:
        original (torch.Tensor): Original audio.
        adversarial (torch.Tensor): Adversarial audio.
        frame_length (int): Length of each frame.
        hop_length (int): Hop length between frames.

    Returns:
        float: Average segmental SNR in dB.
    """
    batch_size = original.shape[0]
    seg_snrs = []

    for b in range(batch_size):
        orig = original[b]
        adv = adversarial[b]

        # Ensure same length
        min_len = min(len(orig), len(adv))
        orig = orig[:min_len]
        adv = adv[:min_len]

        # Calculate noise
        noise = adv - orig

        # Frame-wise processing
        n_frames = 1 + (min_len - frame_length) // hop_length
        frame_snrs = []

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length

            # Extract frame
            orig_frame = orig[start:end]
            noise_frame = noise[start:end]

            # Calculate signal and noise power
            signal_power = torch.mean(orig_frame ** 2)
            noise_power = torch.mean(noise_frame ** 2)

            # Calculate SNR for frame
            if noise_power > 0:
                frame_snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
                # Clipping SNR values as mentioned in the paper
                frame_snr = torch.clamp(frame_snr, min=-10, max=35)
                frame_snrs.append(frame_snr.item())

        # Calculate average SNR for the sample
        if frame_snrs:
            seg_snrs.append(np.mean(frame_snrs))

    # Return average across batch
    return np.mean(seg_snrs) if seg_snrs else 0.0


def calculate_loudness_metric(original, adversarial):
    """
    Calculate the loudness metric (l_dB) as used in the paper.

    Args:
        original (torch.Tensor): Original audio.
        adversarial (torch.Tensor): Adversarial audio.

    Returns:
        float: Loudness metric in dB.
    """
    delta = adversarial - original
    l_dB_delta = 20 * torch.log10(torch.norm(delta) + 1e-10)
    l_dB_original = 20 * torch.log10(torch.norm(original) + 1e-10)

    return (l_dB_delta - l_dB_original).item()