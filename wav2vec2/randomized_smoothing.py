import numpy as np
from collections import defaultdict
from tqdm import tqdm
import evaluate
from utils import load_epsilon_group, get_original_results
import torch

# Initialize metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def add_gaussian_noise(audio: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, audio.shape)
    return audio + noise



def calculate_local_variance(audio: np.ndarray, window_size=1024) -> float:
    """Calculate local variance of audio signal using sliding window"""
    windows = np.lib.stride_tricks.sliding_window_view(audio, window_shape=window_size)
    variances = np.var(windows, axis=1)
    return np.median(variances)

def adaptive_sigma(audio: np.ndarray, base_sigma=0.01, max_sigma=0.05, k=0.5) -> float:
    """
    Compute adaptive sigma based on local signal variance
    Args:
        k: Scaling factor (0.1-1.0), lower = more aggressive adaptation
    """
    local_var = calculate_local_variance(audio)
    scaled_sigma = base_sigma * (1 + k / (local_var + 1e-8))  # Prevent division by zero
    return np.clip(scaled_sigma, base_sigma, max_sigma)

def randomized_smoothing_transcribe(audio_array: np.ndarray, processor, model,
                                   sigma=0.01, num_samples=16):
    """Modified with adaptive sigma"""
    sigma = adaptive_sigma(audio_array, base_sigma=sigma)

    noisy_audios = [add_gaussian_noise(audio_array, sigma) for _ in range(num_samples)]
    transcriptions = []
    batch_size = 4
    for i in range(0, num_samples, batch_size):
        batch = noisy_audios[i:i+batch_size]
        inputs = processor(batch, sampling_rate=16000, return_tensors="pt",
                          padding="longest").input_values.to(model.device)
        with torch.no_grad():
            logits = model(inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch_transcriptions = processor.batch_decode(pred_ids)
            transcriptions.extend(batch_transcriptions)
    word_votes = defaultdict(lambda: defaultdict(int))
    for ts in transcriptions:
        words = ts.split()
        for idx, word in enumerate(words):
            word_votes[idx][word] += 1
    final_transcription = []
    for idx in sorted(word_votes.keys()):
        if idx in word_votes and word_votes[idx]:
            final_word = max(word_votes[idx].items(), key=lambda x: x[1])[0]
            final_transcription.append(final_word)
    return " ".join(final_transcription), sigma  # Return sigma for tracking

def evaluate_randomized_smoothing_with_comparison(model, processor, epsilon_values, alpha_values,
                                                 sigma_values=[0.01], num_samples=16):  # Single default sigma
    results = {}
    for epsilon, alpha in zip(epsilon_values, alpha_values):
        print("\n" + "="*80)
        print(f"EVALUATING EPSILON: {epsilon}, ALPHA: {alpha}")
        print("="*80)
        samples = load_epsilon_group(epsilon, alpha)
        if not samples:
            print(f"No samples found for eps={epsilon}, alpha={alpha}")
            continue
        print(f"Loaded {len(samples)} adversarial samples")
        original_results = get_original_results(samples, processor, model)
        if original_results:
            print("\nORIGINAL RESULTS (NO DEFENSE):")
            print(f"   CER: {original_results['cer']*100:.2f}%")
            print(f"   WER: {original_results['wer']*100:.2f}%")
            print(f"   Samples: {original_results['num_samples']}")
        print("\nTESTING ADAPTIVE SMOOTHING DEFENSE:")
        print("-" * 60)
        results[(epsilon, alpha)] = {
            'original': original_results,
            'defended': {}
        }

        # Track sigma distribution
        sigma_values = []

        for sigma in [0.01]:  # Base sigma for adaptation
            print(f"\nTesting with adaptive sigma...")
            cer_list = []
            wer_list = []
            sigma_list = []
            for sample in tqdm(samples, desc=f"Processing samples"):
                try:
                    defended_transcription, used_sigma = randomized_smoothing_transcribe(
                        sample['audio'], processor, model, sigma=sigma, num_samples=num_samples
                    )
                    sigma_list.append(used_sigma)
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
                avg_sigma = np.mean(sigma_list)
                std_sigma = np.std(sigma_list)
                results[(epsilon, alpha)]['defended']['adaptive'] = {
                    'cer': avg_cer,
                    'wer': avg_wer,
                    'sigma_mean': avg_sigma,
                    'sigma_std': std_sigma,
                    'num_samples': len(cer_list)
                }
                if original_results:
                    cer_improvement = ((original_results['cer'] - avg_cer) / original_results['cer']) * 100 if original_results['cer'] > 0 else 0
                    wer_improvement = ((original_results['wer'] - avg_wer) / original_results['wer']) * 100 if original_results['wer'] > 0 else 0
                    print(f"   Adaptive σ: {avg_sigma:.4f}±{std_sigma:.4f}")
                    print(f"   CER={avg_cer*100:.2f}% (Δ {cer_improvement:+.1f}%), "
                          f"WER={avg_wer*100:.2f}% (Δ {wer_improvement:+.1f}%) (n={len(cer_list)})")
    return results






def print_randomized_smoothing_comprehensive_summary(results):
    print("\n" + "="*100)
    print("COMPREHENSIVE DEFENSE EFFECTIVENESS SUMMARY")
    print("="*100)
    for (epsilon, alpha), result_data in results.items():
        print(f"\nEPSILON: {epsilon}, ALPHA: {alpha}")
        print("-" * 80)
        original = result_data.get('original')
        defended = result_data.get('defended', {})
        if original:
            print(f"Original (No Defense): CER={original['cer']*100:.2f}%, WER={original['wer']*100:.2f}%")
        if defended:
            print(f"Defended Results:")
            print(f"   {'Sigma':<8} {'CER (%)':<10} {'WER (%)':<10} {'CER Δ (%)':<10} {'WER Δ (%)':<10} {'Samples':<8}")
            print("   " + "-" * 50)
            for sigma, metrics in defended.items():
                if original:
                    cer_change = (metrics['cer'] - original['cer'])*100
                    wer_change = (metrics['wer'] - original['wer'])*100
                    print(f"   {sigma:<8} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{cer_change:+10.2f} {wer_change:+10.2f} {metrics['num_samples']:<8}")
                else:
                    print(f"   {sigma:<8} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{'N/A':<10} {'N/A':<10} {metrics['num_samples']:<8}")
        print()

