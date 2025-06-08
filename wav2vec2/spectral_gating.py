#requirements:
#!pip install webrtcvad librosa
#!pip install -U noisereduce


import numpy as np
from collections import defaultdict
import evaluate
import webrtcvad
import noisereduce as nr
from utils import load_epsilon_group, get_original_results, transcribe_audio


# Initialize metrics
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def voice_activity_detection(audio_array: np.ndarray, sample_rate=16000, frame_duration_ms=30):
    """
    Use WebRTC VAD to detect voice activity segments
    Returns a boolean mask indicating voice activity
    """
    # WebRTC VAD requires 16-bit PCM audio
    if audio_array.dtype != np.int16:
        # Convert to int16
        audio_int16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array

    # WebRTC VAD supports specific sample rates
    supported_rates = [8000, 16000, 32000, 48000]
    if sample_rate not in supported_rates:
        print(f"Warning: Sample rate {sample_rate} not supported by WebRTC VAD, using 16000")
        sample_rate = 16000

    vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3 (2 is moderately aggressive)

    frame_length = int(frame_duration_ms * sample_rate / 1000)
    frames = []
    vad_results = []

    # Process audio in frames
    for i in range(0, len(audio_int16), frame_length):
        frame = audio_int16[i:i + frame_length]

        # Pad the last frame if necessary
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant', constant_values=0)

        try:
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            vad_results.extend([is_speech] * frame_length)
        except:
            # If VAD fails, assume it's speech
            vad_results.extend([True] * frame_length)

    # Trim to original length
    vad_mask = np.array(vad_results[:len(audio_array)], dtype=bool)
    return vad_mask

def extract_noise_profile(audio_array: np.ndarray, vad_mask: np.ndarray, sample_rate=16000):
    """
    Extract noise profile from non-speech segments using VAD mask
    """
    # Get non-speech segments
    non_speech_mask = ~vad_mask

    if not np.any(non_speech_mask):
        # If no non-speech segments found, use a small portion from the beginning/end
        print("Warning: No non-speech segments detected, using beginning/end portions")
        segment_length = min(int(0.5 * sample_rate), len(audio_array) // 4)  # 0.5 second or 1/4 of audio
        noise_segments = np.concatenate([
            audio_array[:segment_length],
            audio_array[-segment_length:]
        ])
    else:
        # Extract all non-speech segments
        noise_segments = audio_array[non_speech_mask]

    return noise_segments




def spectral_gating_defense(
    audio_array: np.ndarray,
    sample_rate: int = 16000,
    stationary: bool = False,
    prop_decrease: float = 0.8,
    # NEW: smoothing widths (Hz / ms) instead of bin‐counts
    freq_mask_smooth_hz: float = 500,
    time_mask_smooth_ms: float = 50
) -> np.ndarray:
    """
    Apply spectral gating defense (via noisereduce v2+) to `audio_array`.

    Args:
        audio_array: 1D float32 audio signal in [-1,1].
        sample_rate: Sample rate (Hz).
        stationary:  If True, treat noise as stationary background.
        prop_decrease: 0–1 fraction of noise to remove.
        freq_mask_smooth_hz: width (Hz) of frequency smoothing for the mask.
        time_mask_smooth_ms: width (ms) of time smoothing for the mask.

    Returns:
        cleaned_audio: 1D float32 denoised signal (same length as input).
    """
    # --- 1) Cast / clip to float32 [-1,1] ---
    if audio_array.dtype != np.float32:
        audio_array = audio_array.astype(np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)

    # If there’s a singleton channel dimension (e.g. shape = (1, N) or (N, 1)), squeeze it
    if audio_array.ndim > 1:
        if audio_array.shape[0] == 1:
            audio_array = audio_array.squeeze(0)
        elif audio_array.shape[1] == 1:
            audio_array = audio_array.squeeze(1)

    try:
        # Step 1: Voice Activity Detection (your existing function)
        vad_mask = voice_activity_detection(audio_array, sample_rate)

        # Step 2: Extract noise profile from non‐speech regions
        noise_profile = extract_noise_profile(audio_array, vad_mask, sample_rate)

        # Step 3: Denoise via noisereduce v2+ API
        cleaned_audio = nr.reduce_noise(
            y=audio_array,
            sr=sample_rate,
            y_noise=noise_profile,
            stationary=stationary,
            prop_decrease=prop_decrease,
            freq_mask_smooth_hz=freq_mask_smooth_hz,
            time_mask_smooth_ms=time_mask_smooth_ms
        )

        # Step 4: Make sure we return the exact same length
        if len(cleaned_audio) != len(audio_array):
            if len(cleaned_audio) > len(audio_array):
                cleaned_audio = cleaned_audio[: len(audio_array)]
            else:
                pad_amount = len(audio_array) - len(cleaned_audio)
                cleaned_audio = np.pad(
                    cleaned_audio,
                    (0, pad_amount),
                    mode="constant",
                    constant_values=0
                )

        return cleaned_audio.astype(np.float32)

    except Exception as e:
        print(f"Spectral gating defense failed: {e}")
        # If something breaks, return the original audio unmodified
        return audio_array


def spectral_gating_transcribe(
    audio_array: np.ndarray,
    processor,
    model,
    stationary: bool = False,
    prop_decrease: float = 0.8,
    # NOTE: these names must match spectral_gating_defense
    freq_mask_smooth_hz: float = 500,
    time_mask_smooth_ms: float = 50
):
    """
    Apply spectral gating defense (above) and then transpose it to text.

    Returns:
      (transcription:str, config_name:str)
    """
    defended_audio = spectral_gating_defense(
        audio_array['audio'],
        sample_rate=16000,
        stationary=stationary,
        prop_decrease=prop_decrease,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        time_mask_smooth_ms=time_mask_smooth_ms
    )

    transcription = transcribe_audio(defended_audio, 16000, processor, model)
    cfg_name = (
        f"stat_{stationary}"
        f"_prop_{prop_decrease:.2f}"
        f"_fms_{freq_mask_smooth_hz:.0f}"
        f"_tms_{time_mask_smooth_ms:.0f}"
    )
    return transcription, cfg_name


def evaluate_spectral_gating_with_comparison(
    model, processor, epsilon_values, alpha_values, defense_configs=None
):
    """
    Loops over adversarial samples, applies multiple (updated) spectral‐gating configs,
    and collects CER/WER.  (simplified excerpt)
    """
    if defense_configs is None:
        # NOTE: keys now use 'freq_mask_smooth_hz' & 'time_mask_smooth_ms'
        defense_configs = [
            {'stationary': False, 'prop_decrease': 0.8, 'freq_mask_smooth_hz': 500, 'time_mask_smooth_ms': 50},
            {'stationary': True,  'prop_decrease': 0.8, 'freq_mask_smooth_hz': 500, 'time_mask_smooth_ms': 50},
            {'stationary': False, 'prop_decrease': 0.6, 'freq_mask_smooth_hz': 500, 'time_mask_smooth_ms': 50},
            {'stationary': False, 'prop_decrease': 1.0, 'freq_mask_smooth_hz': 500, 'time_mask_smooth_ms': 50},
            {'stationary': False, 'prop_decrease': 0.8, 'freq_mask_smooth_hz': 200, 'time_mask_smooth_ms': 25},
        ]

    results = {}
    for epsilon, alpha in zip(epsilon_values, alpha_values):
        print("\n" + "=" * 80)
        print(f"EVALUATING EPSILON: {epsilon}, ALPHA: {alpha}")
        print("=" * 80)

        samples = load_epsilon_group(epsilon, alpha)
        if not samples:
            print(f"No samples found for eps={epsilon}, alpha={alpha}")
            continue

        print(f"Loaded {len(samples)} adversarial samples")

        # Compute original (undefended) metrics first

        original_results = get_original_results(samples, processor, model)
        if original_results:
            print("\nORIGINAL RESULTS (NO DEFENSE):")
            print(f"   CER: {original_results['cer']*100:.2f}%")
            print(f"   WER: {original_results['wer']*100:.2f}%")
            print(f"   Samples: {original_results['num_samples']}")


        print("\nTESTING SPECTRAL GATING DEFENSE:")
        print("-" * 60)

        # --- For each defense config, call spectral_gating_transcribe(...) with new args ---
        metrics_list = []
        for cfg in defense_configs:
            stat = cfg['stationary']
            prop = cfg['prop_decrease']
            fms  = cfg['freq_mask_smooth_hz']
            tms  = cfg['time_mask_smooth_ms']

            # Print the human-readable config summary
            print(f"   ├─ stationary={stat}, prop_decrease={prop}, "
                  f"freq_mask_smooth_hz={fms}, time_mask_smooth_ms={tms}")

            # Now actually run defense + transcription on all samples
            defended_results = []
            for wav in samples:
                # Each sample → transcription + cfg_name
                pred, cfg_name = spectral_gating_transcribe(
                    wav, processor, model,
                    stationary=stat,
                    prop_decrease=prop,
                    freq_mask_smooth_hz=fms,
                    time_mask_smooth_ms=tms
                )
                defended_results.append(pred)

            ground_truths = [ sample["ground_truth"] for sample in samples ]

            # Compute CER/WER on this defended_results vs. ground‐truth
            cer = cer_metric.compute(
                predictions=defended_results,
                references=ground_truths
            )
            wer = wer_metric.compute(
                predictions=defended_results,
                references=ground_truths
            )
            metrics_list.append({
                'config':   cfg,
                'cer':      cer,
                'wer':      wer,
                'num_samples': len(samples)
            })

        # Store results under (epsilon, alpha)
        results[(epsilon, alpha)] = {
            'original': original_results,
            'defended': metrics_list
        }

        # Finally, print out each config’s CER/WER
        print("\nSUMMARY for eps={}, alpha={}:".format(epsilon, alpha))
        for m in metrics_list:
            cfg = m['config']
            print(f"   → [stat={cfg['stationary']}, prop={cfg['prop_decrease']}, "
                  f"fms={cfg['freq_mask_smooth_hz']}, tms={cfg['time_mask_smooth_ms']}]  "
                  f"CER={m['cer']*100:.2f}%, WER={m['wer']*100:.2f}%")
        print()

    return results




def print_spectral_gating_comprehensive_summary(results):
    """
    Print a human‐readable summary of CER/WER for each (epsilon, alpha),
    comparing the original (no defense) to each spectral‐gating config.

    Args:
        results: dict with keys (epsilon, alpha) and values:
            {
                'original': {'cer': float, 'wer': float, 'num_samples': int},
                'defended': [
                    {
                        'config': { ...parameters... },
                        'cer': float,
                        'wer': float,
                        'num_samples': int
                    },
                    ...
                ]
            }
    """
    line_width = 80

    for (epsilon, alpha), data in results.items():
        original = data.get('original', {})
        defended_list = data.get('defended', [])

        # Header block
        print("\n" + "=" * line_width)
        print(f"EPSILON: {epsilon:.4f}, ALPHA: {alpha:.4f}")
        print("-" * line_width)

        if original:
            orig_cer = original['cer'] * 100
            orig_wer = original['wer'] * 100
            num_samples = original.get('num_samples', len(defended_list) and defended_list[0]['num_samples'])
            print(f"Original (No Defense): CER={orig_cer:.2f}%, WER={orig_wer:.2f}% (Samples: {num_samples})")
        else:
            print("Original (No Defense): No data available")
        print("\nSpectral Gating Defense Results:")
        print("   Config                          |   CER   |   WER   |  ΔCER   |  ΔWER   | Samples")
        print("   " + "-" * 73)

        # If there are no defended configs, note that and continue
        if not defended_list:
            print("   (no defended configurations found)\n")
            continue

        # Iterate over each defended‐config entry
        for entry in defended_list:
            cfg = entry['config']
            cer_def = entry['cer'] * 100
            wer_def = entry['wer'] * 100
            num_samps = entry.get('num_samples', num_samples)

            # Compute deltas relative to original (if original exists)
            if original:
                delta_cer = cer_def - orig_cer
                delta_wer = wer_def - orig_wer
            else:
                delta_cer = delta_wer = 0.0

            # Build a short “config name” string
            stat_str = "stat" if cfg.get('stationary', False) else "nonstat"
            prop_str = f"p{int(cfg.get('prop_decrease', 0) * 100):02d}"
            fms_str = f"f{int(cfg.get('freq_mask_smooth_hz', 0))}"
            tms_str = f"t{int(cfg.get('time_mask_smooth_ms', 0))}"
            config_name = f"{stat_str}_{prop_str}_{fms_str}_{tms_str}"

            # Format and print a single row
            print(
                f"   {config_name:<30} | "
                f"{cer_def:6.2f}% | "
                f"{wer_def:6.2f}% | "
                f"{delta_cer:7.2f}% | "
                f"{delta_wer:7.2f}% | "
                f"{num_samps:7d}"
            )

        print("\n")  # blank line after each (eps, alpha) block
