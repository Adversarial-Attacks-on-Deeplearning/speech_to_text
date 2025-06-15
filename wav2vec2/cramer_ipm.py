import torch
import torch.nn.functional as F
import numpy as np


def compute_cramer_ipm(original_input, adversarial_input, n_samples=100):
    """
    Compute Cramér-IPM between original and adversarial inputs for a batch.
    Fixed version with robust numerical integration.
    """
    batch_size = original_input.shape[0]
    ipm_values = torch.zeros(batch_size, device=original_input.device)

    for i in range(batch_size):
        # Get flattened distributions
        orig_dist = original_input[i].view(-1)
        adv_dist = adversarial_input[i].view(-1)

        # Use quantiles for better range estimation
        combined = torch.cat([orig_dist, adv_dist])
        min_val = torch.quantile(combined, 0.01)
        max_val = torch.quantile(combined, 0.99)

        # Avoid numerical issues when min_val == max_val
        if torch.abs(max_val - min_val) < 1e-8:
            ipm_values[i] = 0.0
            continue

        # Sample points for integration
        sample_points = torch.linspace(min_val, max_val, n_samples, device=original_input.device)

        # Compute empirical CDFs
        orig_cdf_vals = compute_cdf_at_points(sample_points, orig_dist)
        adv_cdf_vals = compute_cdf_at_points(sample_points, adv_dist)

        # Compute Cramér distance using manual trapezoidal integration
        point_distances = (orig_cdf_vals - adv_cdf_vals) ** 2
        dx = (max_val - min_val) / (n_samples - 1)

        # Manual trapezoidal rule: (f(0) + 2*f(1) + ... + 2*f(n-1) + f(n)) * dx/2
        # Simplified: sum all points * dx (approximation for large n_samples)
        cramer_distance = torch.sum(point_distances) * dx

        ipm_values[i] = cramer_distance

    return ipm_values


def compute_cdf_at_points(points, samples):
    """
    Compute empirical CDF values at specified points.
    Robust vectorized implementation.
    """
    n_samples = len(samples)
    n_points = len(points)

    # Reshape for broadcasting: points [n_points, 1] vs samples [1, n_samples]
    points_expanded = points.unsqueeze(1)  # [n_points, 1]
    samples_expanded = samples.unsqueeze(0)  # [1, n_samples]

    # Count samples <= each point
    comparison = samples_expanded <= points_expanded  # [n_points, n_samples]
    cdf_values = comparison.float().mean(dim=1)  # [n_points]

    return cdf_values


def compute_snr_(original, adversarial):
    """
    Compute SNR between original and adversarial audio.
    More robust version with better numerical stability.
    """
    # Flatten if needed
    if original.dim() > 1:
        original = original.view(-1)
    if adversarial.dim() > 1:
        adversarial = adversarial.view(-1)

    signal_power = torch.mean(original ** 2)
    noise = adversarial - original
    noise_power = torch.mean(noise ** 2)

    # Avoid division by zero and log of zero
    noise_power = torch.clamp(noise_power, min=1e-12)
    signal_power = torch.clamp(signal_power, min=1e-12)

    snr_linear = signal_power / noise_power
    snr_db = 10 * torch.log10(snr_linear)

    return snr_db


def cramer_ipm_attack(audio_tensors, target_transcription, model, processor,
                           epsilon=0.01, sampling_rate=16000, num_iterations=10,
                           lambda_ipm=1.0, n_samples=100, device="cuda",
                           learning_rate=0.01, targeted=True):
    """
    Fixed Cramér-IPM-based attack with robust error handling.
    """
    model.eval()

    # Prepare inputs
    input_values = audio_tensors.to(device)
    if input_values.ndim == 1:
        input_values = input_values.unsqueeze(0)

    batch_size = input_values.shape[0]

    # Tokenize target
    if isinstance(target_transcription, str):
        target_transcription = [target_transcription] * batch_size

    try:
        labels = processor.tokenizer(target_transcription,
                                    return_tensors="pt",
                                    padding=True).input_ids.to(device)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        # Fallback: use simple tokenization
        labels = processor.tokenizer.encode(target_transcription[0], return_tensors="pt").to(device)
        labels = labels.repeat(batch_size, 1)

    # Initialize perturbation
    delta = torch.zeros_like(input_values, requires_grad=True)

    # Use Adam optimizer
    optimizer = torch.optim.Adam([delta], lr=learning_rate)

    # Logging
    snr_values = []
    loss_values = []

    print(f"Starting attack with {num_iterations} iterations...")

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Current adversarial input
        adv_input = input_values + delta
        adv_input = torch.clamp(adv_input, -1.0, 1.0)

        try:
            # Compute CTC loss
            outputs = model(adv_input, labels=labels)
            ctc_loss = outputs.loss

            # Compute Cramér-IPM regularization
            cramer_ipm = compute_cramer_ipm(input_values, adv_input, n_samples)
            cramer_loss = cramer_ipm.mean()

            # Combined loss
            if targeted:
                total_loss = ctc_loss + lambda_ipm * cramer_loss
            else:
                total_loss = -ctc_loss + lambda_ipm * cramer_loss

            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: Invalid loss at iteration {iteration}, skipping...")
                continue

            # Backward pass
            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([delta], max_norm=1.0)

            # Apply gradient step
            optimizer.step()

            # Project perturbation to epsilon ball
            with torch.no_grad():
                # L-infinity projection
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)

                # Ensure final audio is in valid range
                adv_input_projected = torch.clamp(input_values + delta, -1.0, 1.0)
                delta.data = adv_input_projected - input_values

            # Logging every few iterations
            if iteration % 5 == 0 or iteration == num_iterations - 1:
                with torch.no_grad():
                    current_adv = input_values + delta
                    loss_values.append(total_loss.item())

        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            continue

    # Return final adversarial examples
    with torch.no_grad():
        final_adv = torch.clamp(input_values + delta, -1.0, 1.0)


    return final_adv.detach().cpu(), loss_values


# Updated testing function that uses the fixed attack
def test_attack(audio_batch, target_text, model, processor, epsilon=0.01):
    """
    Test function using the fixed attack implementation.
    """
    print(f"Testing fixed attack with epsilon={epsilon}")

    try:
        adversarial_audio, snr_values, loss_values = cramer_ipm_attack(
            audio_tensors=audio_batch,
            target_transcription=target_text,
            model=model,
            processor=processor,
            epsilon=epsilon,
            num_iterations=20,
            lambda_ipm=0.1,
            learning_rate=0.005,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Attack completed successfully!")
        print(f"Final SNR values: {snr_values}")
        print(f"Average SNR: {np.mean(snr_values):.2f} dB")

        return adversarial_audio, snr_values

    except Exception as e:
        print(f"Attack failed with error: {e}")
        return None, None