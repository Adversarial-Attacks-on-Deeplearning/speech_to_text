import torch
from torch.amp import autocast  

def pgd_attack(audio_tensors, target_transcription, model, processor, epsilon=0.3, alpha=0.01, num_iter=40, sampling_rate=16000, device="cuda", convergence_threshold=1e-4):
    """
    Perform PGD attack on Wav2Vec2 model for a batch of audio tensors.

    Args:
        audio_tensors (torch.Tensor): Batch of input audio waveforms (shape: [batch_size, max_length]).
        target_transcription (str): Desired target transcription.
        model (Wav2Vec2ForCTC): Pre-trained Wav2Vec2 model.
        processor (Wav2Vec2Processor): Wav2Vec2 processor.
        epsilon (float): Maximum perturbation magnitude.
        alpha (float): Step size for each iteration.
        num_iter (int): Number of iterations.
        sampling_rate (int): Audio sampling rate.
        device (str): Device to run the model on.
        convergence_threshold (float): Stop if loss change is below this threshold.

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

    # Initialize perturbation
    perturbation = torch.zeros_like(input_values, requires_grad=True, device=device)

    # Iterative PGD attack
    prev_loss = float('inf')
    for i in range(num_iter):
        # Compute input with perturbation
        adv_input = input_values + perturbation

        # Run model outside autocast to avoid FP16 mismatch
        output = model(adv_input, labels=labels)
        loss = output.loss

        # Check for convergence
        if i > 0 and abs(prev_loss - loss.item()) < convergence_threshold:
            print(f"Converged at iteration {i+1}")
            break
        prev_loss = loss.item()

        # Backpropagate
        model.zero_grad()
        loss.backward()

        # Update perturbation in-place with autocast for efficiency
        with torch.no_grad():
            with autocast(device_type='cuda', enabled=True):  # Corrected autocast API
                perturbation.sub_(alpha * torch.sign(perturbation.grad))
                perturbation.clamp_(-epsilon, epsilon)

        # Reset gradient
        perturbation.grad.zero_()

        if i % 10 == 0 and i > 0:
            print(f"Iteration {i+1}/{num_iter}, Loss: {loss.item():.4f}")

    # Create adversarial input
    adversarial_input_values = input_values + perturbation

    return adversarial_input_values.detach().cpu()