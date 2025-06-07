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

    return adversarial_input_values.detach().cpu()
