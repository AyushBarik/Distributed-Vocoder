from diffwave.inference import predict as diffwave_predict
import torch
import numpy as np
import IPython.display as ipd
import torchaudio


import soundfile as sf
import librosa
import warnings
from pymcd.mcd import Calculate_MCD
from pystoi import stoi
from pesq import pesq

import time
import GPUtil

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Remove this - just use default GPU 0

model_dir = 'diffwave-ljspeech-22kHz-1000578.pt'
# Load 240s mel spectrogram for testing
spectrogram_data = torch.from_numpy(np.load('audio/240s_mel.npy')).float().unsqueeze(0).to('cuda')
spectrogram = spectrogram_data

print(f"Loading 240s mel spectrogram with shape: {spectrogram.shape}")
print(f"Expected audio duration: ~240 seconds")

# Add RTF timing - only around the actual inference
print("Starting DiffWave inference for 240s clip...")
start_time = time.time()
audio, sample_rate = diffwave_predict(spectrogram, model_dir, fast_sampling=True)
torch.cuda.synchronize()  # Wait for GPU to finish before stopping timer
end_time = time.time()
inference_time = end_time - start_time

print(f"Inference completed! Time taken: {inference_time:.3f} seconds")

audio_for_playback = audio.squeeze().cpu()
ipd.display(ipd.Audio(audio_for_playback, rate=sample_rate))
torchaudio.save("generated_240s_test.wav", audio_for_playback.unsqueeze(0), sample_rate)

# Use 240s original audio for comparison
original_file = "audio/240s.wav"
generated_file = "generated_240s_test.wav"

print("--- Running Vocoder Benchmarks ---")
print(f"Original: {original_file}")
print(f"Generated: {generated_file}\n")


# --- Metric 1: Mel-Cepstral Distortion (MCD) ↓ ---
# Lower is better. Checks spectral accuracy.
mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
mcd_value = mcd_toolbox.calculate_mcd(original_file, generated_file)
print(f"MCD↓:  {mcd_value:.2f} dB")


# --- Metric 2: Short-Time Objective Intelligibility (STOI) ↑ ---
original_audio, sr = sf.read(original_file)
generated_audio, sr_gen = sf.read(generated_file)
if sr != sr_gen:
    generated_audio = librosa.resample(y=generated_audio, orig_sr=sr_gen, target_sr=sr)

# --- THIS IS THE FIX ---
# Trim both audio files to the length of the shorter one
min_len = min(len(original_audio), len(generated_audio))
original_audio = original_audio[:min_len]
generated_audio = generated_audio[:min_len]
# --- END OF FIX ---

stoi_score = stoi(original_audio, generated_audio, sr, extended=False)
print(f"STOI↑:  {stoi_score:.4f}")


# --- Metric 3: Wideband PESQ (WB-PESQ) ↑ ---
# Higher is better (~1-4.5). Checks overall perceptual quality.
# Note: WB-PESQ MUST use 16kHz audio. We resample to meet this requirement.
sr_pesq = 16000
original_16k = librosa.resample(y=original_audio, orig_sr=sr, target_sr=sr_pesq)
generated_16k = librosa.resample(y=generated_audio, orig_sr=sr, target_sr=sr_pesq)
pesq_score = pesq(sr_pesq, original_16k, generated_16k, 'wb')
print(f"PESQ↑:  {pesq_score:.2f}")

# Performance Metrics (using existing audio from cell 1)
print("\n--- Performance Metrics ---")

# Get model size/parameters
checkpoint = torch.load(model_dir, map_location='cpu')
if isinstance(checkpoint, dict):
    if 'model' in checkpoint:
        model_state = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint
    param_count = sum(param.numel() for param in model_state.values())
else:
    param_count = sum(p.numel() for p in checkpoint.parameters())

print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")
del checkpoint
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use the audio you ALREADY generated in cell 1
audio_duration = len(audio_for_playback) / sample_rate
print(f"Audio duration: {audio_duration:.3f} seconds")

# RTF calculation (FIXED)
rtf = inference_time / audio_duration
print(f"Inference time: {inference_time:.3f} seconds")
print(f"RTF: {rtf:.3f}")

# Debug the RTF calculation (FIXED - using correct variables)
print("=== Debugging RTF calculation ===")

# Check the audio details (using YOUR actual variables)
print(f"Audio shape: {audio.shape}")
print(f"Sample rate: {sample_rate}")
print(f"Audio length (samples): {len(audio_for_playback)}")

# Calculate duration manually
audio_duration_debug = len(audio_for_playback) / sample_rate
print(f"Audio duration: {audio_duration_debug:.3f} seconds")

# RTF calculation
print(f"RTF = {inference_time:.3f} / {audio_duration_debug:.3f} = {rtf:.3f}")

# Also check the original audio for comparison
original_audio_check, original_sr = torchaudio.load(original_file)
original_duration = len(original_audio_check.squeeze()) / original_sr
print(f"Original audio duration: {original_duration:.3f} seconds") 