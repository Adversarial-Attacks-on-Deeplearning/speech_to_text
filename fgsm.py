import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def fgsm_attack_single_audio(model, processor, waveform, target_text, epsilon):
    """
    Perform FGSM attack on a single audio waveform for a Wav2Vec2 ASR model.

    Args:
        model (torch.nn.Module): Pre-trained Wav2Vec2ForCTC model.
        processor (Wav2Vec2Processor): Processor to handle audio and text.
        waveform (torch.Tensor): Input waveform of shape (T,) or (1, T) with values in [-1, 1].
        target_text (str): The target transcription.
        epsilon (float): Perturbation magnitude.

    Returns:
        torch.Tensor: Adversarial waveform with shape (T,) in the range [-1, 1].
    """
    device = next(model.parameters()).device

    # Ensure waveform has shape [1, T]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # shape: [1, T]

    # Process the audio input with explicit sampling_rate
    inputs = processor(waveform.squeeze().cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)  # shape: [1, T']

    # Process the target text by directly using the underlying tokenizer
    target = processor.tokenizer(target_text, return_tensors="pt").input_ids.to(device)

    # Forward pass to get logits and determine input length for CTC loss
    with torch.no_grad():
        logits = model(input_values).logits  # shape: [1, T', vocab_size]
    logit_length = logits.size(1)
    input_lengths = torch.full((1,), logit_length, dtype=torch.long, device=device)
    target_lengths = torch.full((1,), target.size(1), dtype=torch.long, device=device)

    # Set up the CTC loss (using the pad_token_id as blank)
    ctc_loss_fn = torch.nn.CTCLoss(blank=processor.tokenizer.pad_token_id, zero_infinity=True)

    # Enable gradient tracking for input_values
    input_values.requires_grad = True

    # Forward pass with gradient tracking
    logits = model(input_values).logits
    # CTC loss expects logits in shape (T, N, C)
    logits_for_loss = logits.transpose(0, 1)  # shape: [T', 1, vocab_size]
    loss = ctc_loss_fn(logits_for_loss.log_softmax(dim=-1), target, input_lengths, target_lengths)

    # Backward pass to compute gradients with respect to input
    model.zero_grad()
    loss.backward()

    # Compute the sign of the gradient and add the perturbation
    grad_sign = torch.sign(input_values.grad.data)
    adv_input = input_values + epsilon * grad_sign
    adv_input = torch.clamp(adv_input, -1.0, 1.0)

    # Return the adversarial waveform (removing the batch dimension)
    return adv_input.detach().cpu().squeeze()


