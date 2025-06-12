# requirments: 
#!pip install pydub
#!sudo apt-get install ffmpeg  # On Colab or Linux


import numpy as np
import os
import tempfile
from tqdm import tqdm
import evaluate
from pydub import AudioSegment
from utils import load_epsilon_group, get_original_results, transcribe_audio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache


# Initialize metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

# Thread-local storage for metrics (since evaluate metrics may not be thread-safe)
thread_local = threading.local()

def get_thread_metrics():
    """Get thread-local metrics instances"""
    if not hasattr(thread_local, 'cer_metric'):
        thread_local.cer_metric = evaluate.load("cer")
        thread_local.wer_metric = evaluate.load("wer")
    return thread_local.cer_metric, thread_local.wer_metric


def mp3_compress_defense_optimized(audio_array: np.ndarray, sample_rate=16000, bitrate="128k") -> np.ndarray:
    """Optimized MP3 compression/decompression defense with memory improvements"""

    # Ensure audio is in correct range and format
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)

    # Clip to prevent overflow
    audio_array = np.clip(audio_array, -1.0, 1.0)

    # Convert to 16-bit PCM for MP3 encoding
    audio_int16 = (audio_array * 32767).astype(np.int16)

    # Handle potential channel dimension issues
    if len(audio_int16.shape) > 1:
        if audio_int16.shape[0] == 1:
            audio_int16 = audio_int16.squeeze(0)
        elif audio_int16.shape[1] == 1:
            audio_int16 = audio_int16.squeeze(1)

    try:
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )

        # Use in-memory compression when possible (faster than file I/O)
        from io import BytesIO
        
        # Export to memory buffer
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate=bitrate)
        mp3_buffer.seek(0)
        
        # Read back from memory buffer
        compressed_audio = AudioSegment.from_file(mp3_buffer, format="mp3")
        mp3_buffer.close()

        # Convert back to numpy array
        audio_samples = np.array(compressed_audio.get_array_of_samples(), dtype=np.float32)

        # Normalize back to [-1, 1] range
        audio_normalized = audio_samples / 32767.0

        # Ensure output length matches input (pad or trim if necessary)
        input_len = len(audio_array)
        output_len = len(audio_normalized)
        
        if output_len != input_len:
            if output_len > input_len:
                audio_normalized = audio_normalized[:input_len]
            else:
                # Use zeros for padding (more efficient than np.pad for simple constant padding)
                padding = input_len - output_len
                audio_normalized = np.concatenate([audio_normalized, np.zeros(padding, dtype=np.float32)])

        return audio_normalized

    except Exception as e:
        print(f"MP3 compression failed: {e}")
        # Return original audio if compression fails
        return audio_array


def process_single_sample(sample, processor, model, bitrate):
    """Process a single sample with MP3 compression defense"""
    try:
        # Apply defense
        defended_audio = mp3_compress_defense_optimized(sample['audio'], sample_rate=16000, bitrate=bitrate)
        
        # Transcribe
        defended_transcription = transcribe_audio(defended_audio, 16000, processor, model)
        
        # Get thread-local metrics
        cer_metric_local, wer_metric_local = get_thread_metrics()
        
        # Calculate metrics
        ground_truth = sample['ground_truth']
        cer = cer_metric_local.compute(predictions=[defended_transcription], references=[ground_truth])
        wer = wer_metric_local.compute(predictions=[defended_transcription], references=[ground_truth])
        
        return {'cer': cer, 'wer': wer, 'success': True}
        
    except Exception as e:
        return {'error': str(e), 'success': False}


def mp3_compression_transcribe_batch(samples, processor, model, bitrate="128k", max_workers=4):
    """Process multiple samples in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(process_single_sample, sample, processor, model, bitrate): i 
            for i, sample in enumerate(samples)
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_sample), total=len(samples), desc=f"Processing with {bitrate}"):
            result = future.result()
            if result['success']:
                results.append(result)
            else:
                print(f"Error processing sample: {result.get('error', 'Unknown error')}")
    
    return results


def evaluate_mp3_compression_with_comparison(model, processor, epsilon_values, alpha_values,
                                                     bitrate_values=["128k", "96k", "64k", "32k"],
                                                     max_workers=2):
    """Optimized evaluation with parallel processing"""
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

        # Get original (undefended) results
        original_results = get_original_results(samples, processor, model)
        if original_results:
            print("\nORIGINAL RESULTS (NO DEFENSE):")
            print(f"   CER: {original_results['cer']*100:.2f}%")
            print(f"   WER: {original_results['wer']*100:.2f}%")
            print(f"   Samples: {original_results['num_samples']}")

        print("\nTESTING MP3 COMPRESSION DEFENSE:")
        print("-" * 60)

        results[(epsilon, alpha)] = {
            'original': original_results,
            'defended': {}
        }

        # Test different bitrates with parallel processing
        for bitrate in bitrate_values:
            print(f"\nTesting with bitrate: {bitrate}")
            
            # Process samples in parallel
            batch_results = mp3_compression_transcribe_batch(
                samples, processor, model, bitrate=bitrate, max_workers=max_workers
            )
            
            if batch_results:
                # Extract CER and WER values
                cer_list = [r['cer'] for r in batch_results]
                wer_list = [r['wer'] for r in batch_results]
                
                avg_cer = np.mean(cer_list)
                avg_wer = np.mean(wer_list)

                results[(epsilon, alpha)]['defended'][bitrate] = {
                    'cer': avg_cer,
                    'wer': avg_wer,
                    'num_samples': len(batch_results)
                }

                if original_results:
                    cer_improvement = ((original_results['cer'] - avg_cer) / original_results['cer']) * 100 if original_results['cer'] > 0 else 0
                    wer_improvement = ((original_results['wer'] - avg_wer) / original_results['wer']) * 100 if original_results['wer'] > 0 else 0

                    print(f"   Bitrate {bitrate}:")
                    print(f"   CER={avg_cer*100:.2f}% (Δ {cer_improvement:+.1f}%), "
                          f"WER={avg_wer*100:.2f}% (Δ {wer_improvement:+.1f}%) (n={len(batch_results)})")
                else:
                    print(f"   Bitrate {bitrate}:")
                    print(f"   CER={avg_cer*100:.2f}%, WER={avg_wer*100:.2f}% (n={len(batch_results)})")

    return results


# Keep original functions for backward compatibility
def mp3_compress_defense(audio_array: np.ndarray, sample_rate=16000, bitrate="128k") -> np.ndarray:
    """Original MP3 compression defense - kept for compatibility"""
    return mp3_compress_defense_optimized(audio_array, sample_rate, bitrate)


def mp3_compression_transcribe(audio_array: np.ndarray, processor, model, bitrate="128k"):
    """Apply MP3 compression defense and transcribe - kept for compatibility"""
    defended_audio = mp3_compress_defense_optimized(audio_array, sample_rate=16000, bitrate=bitrate)
    transcription = transcribe_audio(defended_audio, 16000, processor, model)
    return transcription, bitrate



def print_mp3_compress_comprehensive_summary(results):
    """Print comprehensive summary - unchanged"""
    print("\n" + "="*100)
    print("COMPREHENSIVE MP3 COMPRESSION DEFENSE EFFECTIVENESS SUMMARY")
    print("="*100)

    for (epsilon, alpha), result_data in results.items():
        print(f"\nEPSILON: {epsilon}, ALPHA: {alpha}")
        print("-" * 80)

        original = result_data.get('original')
        defended = result_data.get('defended', {})

        if original:
            print(f"Original (No Defense): CER={original['cer']*100:.2f}%, WER={original['wer']*100:.2f}%")

        if defended:
            print(f"MP3 Compression Defense Results:")
            print(f"   {'Bitrate':<8} {'CER (%)':<10} {'WER (%)':<10} {'CER Δ (%)':<10} {'WER Δ (%)':<10} {'Samples':<8}")
            print("   " + "-" * 58)

            for bitrate, metrics in defended.items():
                if original:
                    cer_change = (metrics['cer'] - original['cer'])*100
                    wer_change = (metrics['wer'] - original['wer'])*100
                    print(f"   {bitrate:<8} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{cer_change:+10.2f} {wer_change:+10.2f} {metrics['num_samples']:<8}")
                else:
                    print(f"   {bitrate:<8} {metrics['cer']*100:<10.2f} {metrics['wer']*100:<10.2f} "
                          f"{'N/A':<10} {'N/A':<10} {metrics['num_samples']:<8}")
        print()