import numpy as np
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from utils import load_epsilon_group, get_original_results, transcribe_audio

# For FX quantization config
qconfig_dict = {
    "": torch.quantization.default_dynamic_qconfig
}

import torch.quantization.quantize_fx as quantize_fx
from collections import defaultdict
import json
from pathlib import Path
import torchaudio
from tqdm import tqdm
import evaluate

# Initialize metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def get_device():
    """Get the appropriate device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def move_to_device(data, device):
    """Move data to the specified device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    else:
        return data

def input_quantization_defense(audio_array: np.ndarray, bit_width: int = 8, device=None) -> np.ndarray:
    """
    Apply input quantization as preprocessing defense

    Args:
        audio_array: Input audio array
        bit_width: Quantization bit width (4, 8, 16)
        device: Device to perform operations on
    """
    if device is None:
        device = get_device()

    # Convert to tensor and move to device
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float().to(device)
    else:
        audio_tensor = audio_array.float().to(device)

    # Ensure audio is in [-1, 1] range
    audio_tensor = torch.clamp(audio_tensor, -1.0, 1.0)

    # Calculate quantization levels
    num_levels = 2 ** bit_width

    # Quantize the input
    # Scale to [0, num_levels-1]
    scaled = (audio_tensor + 1) / 2 * (num_levels - 1)
    quantized = torch.round(scaled)

    # Scale back to [-1, 1]
    dequantized = (quantized / (num_levels - 1)) * 2 - 1

    # Move back to CPU and convert to numpy if needed
    result = dequantized.cpu().numpy()
    return result

def apply_dynamic_quantization(model, device=None):
    """
    Apply dynamic quantization to the model

    Args:
        model: The model to quantize
        device: Device to perform operations on
    """
    if device is None:
        device = get_device()

    try:
        # Move model to CPU for quantization (PyTorch requirement)
        model_cpu = model.cpu()

        # Apply dynamic quantization to Linear layers
        quantized_model = quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)


        print("Successfully applied dynamic quantization")
        return quantized_model

    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
        print("Returning original model")
        return model.to(device)


def input_quantization_transcribe(audio_array: np.ndarray, processor, model, bit_width: int = 8, device=None):
    """Apply input quantization defense and transcribe"""
    if device is None:
        device = get_device()

    defended_audio = input_quantization_defense(audio_array, bit_width=bit_width, device=device)
    transcription = transcribe_audio(defended_audio, 16000, processor, model)
    return transcription, bit_width

def model_quantization_transcribe(audio_array: np.ndarray, processor, quantized_model, quantization_type: str, device=None):
    """Transcribe using quantized model"""
    if device is None:
        device = get_device()

    # Move audio to device
    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float().to(device)
    else:
        audio_tensor = audio_array.float().to(device)

    # For quantized models, we may need to use CPU
    if quantization_type in ["dynamic", "static"]:
        audio_tensor = audio_tensor.cpu()
        quantized_model = quantized_model.cpu()

    # Process the audio
    try:
        inputs = processor(audio_tensor.numpy() if isinstance(audio_tensor, torch.Tensor) else audio_tensor,
                          sampling_rate=16000, return_tensors="pt")

        if quantization_type in ["dynamic", "static"]:
            input_values = inputs.input_values.cpu()
        else:
            input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = quantized_model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        return transcription.lower().strip()

    except Exception as e:
        print(f"Error in model quantization transcribe: {e}")
        # Fallback to original transcription method
        return transcribe_audio(audio_array, 16000, processor, quantized_model)

def evaluate_quantization_defense_comprehensive(model, processor, epsilon_values, alpha_values,
                                              input_bit_widths=[4, 8, 16]):
    """
    Comprehensive evaluation of quantization defenses

    Args:
        model: Original model
        processor: Audio processor
        epsilon_values: List of epsilon values to test
        alpha_values: List of alpha values to test
        input_bit_widths: List of input quantization bit widths
    """
    device = get_device()
    results = {}

    # Prepare quantized models once
    quantized_models = {}

    # Get example input for static quantization calibration
    sample_audio = np.random.randn(16000).astype(np.float32)  # 1 second of audio
    example_inputs = processor(sample_audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)



    for epsilon, alpha in zip(epsilon_values, alpha_values):
        print("\n" + "="*80)
        print(f"EVALUATING EPSILON: {epsilon}, ALPHA: {alpha}")
        print("="*80)

        samples = load_epsilon_group(epsilon, alpha)
        if not samples:
            print(f"No samples found for eps={epsilon}, alpha={alpha}")
            continue

        print(f"Loaded {len(samples)} adversarial samples")

        # Get original (undefended) results
        original_results = get_original_results(samples, processor, model)
        if original_results:
            print("\nORIGINAL RESULTS (NO DEFENSE):")
            print(f"   CER: {original_results['cer']*100:.2f}%")
            print(f"   WER: {original_results['wer']*100:.2f}%")
            print(f"   Samples: {original_results['num_samples']}")

        results[(epsilon, alpha)] = {
            'original': original_results,
            'input_quantization': {},
            'model_quantization': {},
        }

        # Test input quantization
        print("\nTESTING INPUT QUANTIZATION DEFENSE:")
        print("-" * 60)

        for bit_width in input_bit_widths:
            print(f"\nTesting input quantization with {bit_width}-bit")
            cer_list = []
            wer_list = []

            for sample in tqdm(samples, desc=f"Processing {bit_width}-bit input"):
                try:
                    defended_transcription, used_bit_width = input_quantization_transcribe(
                        sample['audio'], processor, model, bit_width=bit_width, device=device
                    )

                    ground_truth = sample['ground_truth']
                    cer = cer_metric.compute(predictions=[defended_transcription], references=[ground_truth])
                    wer = wer_metric.compute(predictions=[defended_transcription], references=[ground_truth])

                    cer_list.append(cer)
                    wer_list.append(wer)

                except Exception as e:
                    print(f"Error processing sample: {str(e)}")
                    continue

            if cer_list and wer_list:
                avg_cer = np.mean(cer_list)
                avg_wer = np.mean(wer_list)

                results[(epsilon, alpha)]['input_quantization'][f"{bit_width}bit"] = {
                    'cer': avg_cer,
                    'wer': avg_wer,
                    'num_samples': len(cer_list)
                }

                if original_results:
                    cer_improvement = ((original_results['cer'] - avg_cer) / original_results['cer']) * 100 if original_results['cer'] > 0 else 0
                    wer_improvement = ((original_results['wer'] - avg_wer) / original_results['wer']) * 100 if original_results['wer'] > 0 else 0

                    print(f"   {bit_width}-bit: CER={avg_cer*100:.2f}% (Δ {cer_improvement:+.1f}%), "
                          f"WER={avg_wer*100:.2f}% (Δ {wer_improvement:+.1f}%) (n={len(cer_list)})")

    return results

def print_quantization_comprehensive_summary(results):
    """Print comprehensive summary of quantization defense results"""
    print("\n" + "="*120)
    print("COMPREHENSIVE QUANTIZATION DEFENSE EFFECTIVENESS SUMMARY")
    print("="*120)

    for (epsilon, alpha), result_data in results.items():
        print(f"\nEPSILON: {epsilon}, ALPHA: {alpha}")
        print("-" * 100)

        original = result_data.get('original')
        input_quant = result_data.get('input_quantization', {})

        if original:
            print(f"Original (No Defense): CER={original['cer']*100:.2f}%, WER={original['wer']*100:.2f}%")

        # Input Quantization Results
        if input_quant:
            print(f"\nInput Quantization Defense Results:")
            print(f"   {'Method':<12} {'CER (%)':<10} {'WER (%)':<10} {'CER Δ (%)':<10} {'WER Δ (%)':<10} {'Samples':<8}")
            print("   " + "-" * 68)

            for method, metrics in input_quant.items():
                if original:
                    cer_change = (metrics['cer'] - original['cer'])*100
                    wer_change = (metrics['wer'] - original['wer'])*100
                    print(f"   {method:<12} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{cer_change:+10.2f} {wer_change:+10.2f} {metrics['num_samples']:<8}")
                else:
                    print(f"   {method:<12} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{'N/A':<10} {'N/A':<10} {metrics['num_samples']:<8}")


        print()

# Usage example function
def run_quantization_defense_evaluation(model, processor, epsilon_values, alpha_values):
    """
    Main function to run the comprehensive quantization defense evaluation

    Args:
        model: The pre-trained wav2vec2 model
        processor: The audio processor
        epsilon_values: List of epsilon values for adversarial attacks
        alpha_values: List of alpha values for adversarial attacks
    """
    print("Starting Quantization Defense Evaluation...")
    print(f"Testing epsilon values: {epsilon_values}")
    print(f"Testing alpha values: {alpha_values}")

    # Run comprehensive evaluation
    results = evaluate_quantization_defense_comprehensive(
        model=model,
        processor=processor,
        epsilon_values=epsilon_values,
        alpha_values=alpha_values,
        input_bit_widths=[4, 8, 16],
    )

    # Print comprehensive summary
    print_quantization_comprehensive_summary(results)

    return results
