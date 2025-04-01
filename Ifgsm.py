import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def iterative_untargeted_fgsm_attack_single_audio(model, processor, waveform, true_text, epsilon, num_steps=10):
    """
    Perform an iterative untargeted FGSM attack on a single audio waveform for a Wav2Vec2 ASR model.

    This attack increases the CTC loss for the ground-truth transcription (true_text) so that the
    model is more likely to mis-transcribe the input.

    Args:
        model (torch.nn.Module): Pre-trained Wav2Vec2ForCTC model.
        processor (Wav2Vec2Processor): Processor to handle audio and text.
        waveform (torch.Tensor): Input waveform of shape (T,) or (1, T) with values in [-1, 1].
        true_text (str): The ground-truth transcription.
        epsilon (float): Maximum total perturbation magnitude (L∞ norm).
        num_steps (int): Number of iterations for the attack.

    Returns:
        torch.Tensor: Adversarial waveform with shape (T,) in the range [-1, 1].
    """
    device = next(model.parameters()).device

    # Ensure waveform has shape [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # shape: [1, T]

    # Process the audio input with explicit sampling_rate
    inputs = processor(waveform.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    original_input = inputs.input_values.to(device)  # shape: [1, T']

    # Process the ground-truth text with the underlying tokenizer
    true_label = processor.tokenizer(true_text, return_tensors="pt").input_ids.to(device)

    # Forward pass to determine logit length for the CTC loss
    with torch.no_grad():
        logits = model(original_input).logits  # shape: [1, T', vocab_size]
    logit_length = logits.size(1)
    input_lengths = torch.full((1,), logit_length, dtype=torch.long, device=device)
    true_label_lengths = torch.full((1,), true_label.size(1), dtype=torch.long, device=device)

    # Set up the CTC loss (using the pad_token_id as blank)
    ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    # Initialize adversarial input with the original input
    adv_input = original_input.clone()

    # Calculate step size (alpha) per iteration
    alpha = epsilon / num_steps

    for _ in range(num_steps):
        adv_input.requires_grad = True

        # Forward pass with current adversarial input
        logits = model(adv_input).logits
        logits_for_loss = logits.transpose(0, 1)  # shape: [T', 1, vocab_size]
        loss = ctc_loss_fn(logits_for_loss.log_softmax(dim=-1), true_label, input_lengths, true_label_lengths)

        # Backward pass to compute gradients
        model.zero_grad()
        loss.backward()
        grad_sign = torch.sign(adv_input.grad.data)

        # Update adversarial input by adding a small step in the direction that increases the loss
        adv_input = adv_input + alpha * grad_sign

        # Project the total perturbation to remain within the epsilon L∞ ball
        perturbation = torch.clamp(adv_input - original_input, min=-epsilon, max=epsilon)
        adv_input = original_input + perturbation

        # Ensure waveform values remain in the valid range [-1, 1]
        adv_input = torch.clamp(adv_input, -1.0, 1.0)

        # Detach to avoid gradient accumulation in subsequent iterations
        adv_input = adv_input.detach()

    # Return the adversarial waveform (removing the batch dimension)
    return adv_input.cpu().squeeze()

