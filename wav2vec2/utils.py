import torch

def transcribe_audio(audio_array, sampling_rate, processor, model):
    if sampling_rate != 16000:
        raise ValueError(f"Expected 16kHz audio but got {sampling_rate}Hz")

    # Pass sampling_rate to the processor
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="longest")
    input_values = inputs.input_values.to("cuda")

    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

    return transcription
