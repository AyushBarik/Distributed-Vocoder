#!/usr/bin/env python3
"""
Smart Phoneme-Based Chunked Inference
Finds silence regions in mel spectrogram for natural chunk boundaries
"""

from diffwave.inference import predict as diffwave_predict
import torch
import numpy as np
import torchaudio
import soundfile as sf
import librosa
from pymcd.mcd import Calculate_MCD
from pystoi import stoi
from pesq import pesq
import time

def find_silence_boundaries(mel_spectrogram, silence_threshold_db=-40, min_silence_frames=5):
    """
    Find silence regions in mel spectrogram for smart chunking boundaries
    
    Args:
        mel_spectrogram: [batch, n_mels, time_frames]
        silence_threshold_db: dB threshold for silence detection
        min_silence_frames: Minimum consecutive silent frames to consider as boundary
    
    Returns:
        List of frame indices where silence regions occur (good chunk boundaries)
    """
    # DiffWave mel spectrograms are already in dB scale and normalized to [0,1]
    # where 0 = -100dB and 1 = 0dB, so we can work directly with these values
    energy_per_frame = torch.mean(mel_spectrogram, dim=1).squeeze()  # Average energy across mel bins
    # Convert normalized values back to approximate dB scale for thresholding
    energy_db = (energy_per_frame * 100) - 100  # Convert [0,1] back to [-100,0] dB range
    
    # Find silent frames
    silent_frames = energy_db < silence_threshold_db
    
    # Find silence regions (consecutive silent frames)
    silence_boundaries = []
    in_silence = False
    silence_start = 0
    
    for i, is_silent in enumerate(silent_frames):
        if is_silent and not in_silence:
            # Start of silence region
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            # End of silence region
            silence_length = i - silence_start
            if silence_length >= min_silence_frames:
                # Use middle of silence region as boundary
                boundary = silence_start + silence_length // 2
                silence_boundaries.append(boundary)
            in_silence = False
    
    # FIX: Handle case where spectrogram ends in silence
    if in_silence:
        silence_length = len(silent_frames) - silence_start
        if silence_length >= min_silence_frames:
            boundary = silence_start + silence_length // 2
            silence_boundaries.append(boundary)
    
    # Always include start and end
    if 0 not in silence_boundaries:
        silence_boundaries.insert(0, 0)
    if len(energy_per_frame) - 1 not in silence_boundaries:
        silence_boundaries.append(len(energy_per_frame))  # Use len() not len()-1 for proper slicing
    
    return sorted(silence_boundaries)

def smart_chunked_predict(mel_spectrogram, model_dir, max_chunk_frames=512, fast_sampling=True):
    """
    Smart chunked inference using silence boundaries for natural phoneme chunking
    """
    print("=== Smart Phoneme-Based Chunked Inference ===")
    print(f"Input mel shape: {mel_spectrogram.shape}")
    
    # Find natural silence boundaries
    boundaries = find_silence_boundaries(mel_spectrogram, silence_threshold_db=-40, min_silence_frames=5)
    
    # FIX: Create chunks between boundaries with proper logic
    chunks = []
    
    for i in range(len(boundaries) - 1):
        chunk_start = boundaries[i]
        chunk_end = boundaries[i + 1]
        
        # FIX: Skip zero-length chunks
        if chunk_start >= chunk_end:
            continue
            
        chunk_size = chunk_end - chunk_start
        
        if chunk_size <= max_chunk_frames:
            # Chunk fits within limit
            chunks.append((chunk_start, chunk_end))
        else:
            # Chunk too big, split it at max size
            current_start = chunk_start
            while current_start + max_chunk_frames < chunk_end:
                chunks.append((current_start, current_start + max_chunk_frames))
                current_start += max_chunk_frames
            # Add remaining part (if any)
            if current_start < chunk_end:
                chunks.append((current_start, chunk_end))
    
    print(f"Found {len(boundaries)} silence boundaries")
    print(f"Created {len(chunks)} smart chunks")  # FIX: Log chunks not boundaries
    
    # Process each chunk
    audio_segments = []
    sample_rate = None
    
    start_time = time.time()
    
    for i, (start_frame, end_frame) in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}: frames {start_frame}-{end_frame}")
        
        # Extract chunk
        mel_chunk = mel_spectrogram[:, :, start_frame:end_frame]
        
        # Generate audio
        audio_chunk, sr = diffwave_predict(mel_chunk, model_dir, fast_sampling=fast_sampling)
        # FIX: Remove unnecessary per-chunk synchronize
        
        if sample_rate is None:
            sample_rate = sr
            
        audio_segments.append(audio_chunk.squeeze().cpu())
        
        # Free memory (remove empty_cache unless needed)
        del mel_chunk, audio_chunk
    
    # FIX: Single synchronize at end for timing accuracy
    torch.cuda.synchronize()
    
    # Concatenate all chunks (no overlap-add needed since we used silence boundaries)
    combined_audio = torch.cat(audio_segments, dim=0)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.3f}s")
    print(f"Final audio shape: {combined_audio.shape}")
    
    return combined_audio.unsqueeze(0), sample_rate

# Main execution - exactly like your inference.py
if __name__ == "__main__":
    model_dir = 'diffwave-ljspeech-22kHz-1000578.pt'
    spectrogram_data = torch.from_numpy(np.load('mel_spectrogram_data.npy')).float().unsqueeze(0).to('cuda')
    spectrogram = spectrogram_data

    # Smart chunked inference
    start_time = time.time()
    audio, sample_rate = smart_chunked_predict(spectrogram, model_dir, max_chunk_frames=512, fast_sampling=True)
    torch.cuda.synchronize()
    end_time = time.time()
    inference_time = end_time - start_time

    audio_for_playback = audio.squeeze().cpu()
    torchaudio.save("chunked_diffwave.wav", audio_for_playback.unsqueeze(0), sample_rate)

    # Benchmarks
    original_file = "example.wav"
    generated_file = "chunked_diffwave.wav"

    print("\n--- Running Vocoder Benchmarks ---")
    print(f"Original: {original_file}")
    print(f"Generated: {generated_file}\n")

    # MCD
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    mcd_value = mcd_toolbox.calculate_mcd(original_file, generated_file)
    print(f"MCD↓:  {mcd_value:.2f} dB")

    # STOI
    original_audio, sr = sf.read(original_file)
    generated_audio, sr_gen = sf.read(generated_file)
    if sr != sr_gen:
        generated_audio = librosa.resample(y=generated_audio, orig_sr=sr_gen, target_sr=sr)

    min_len = min(len(original_audio), len(generated_audio))
    original_audio = original_audio[:min_len]
    generated_audio = generated_audio[:min_len]

    stoi_score = stoi(original_audio, generated_audio, sr, extended=False)
    print(f"STOI↑:  {stoi_score:.4f}")

    # PESQ
    sr_pesq = 16000
    original_16k = librosa.resample(y=original_audio, orig_sr=sr, target_sr=sr_pesq)
    generated_16k = librosa.resample(y=generated_audio, orig_sr=sr, target_sr=sr_pesq)
    pesq_score = pesq(sr_pesq, original_16k, generated_16k, 'wb')
    print(f"PESQ↑:  {pesq_score:.2f}")

    # Performance metrics
    print("\n--- Performance Metrics ---")
    audio_duration = len(audio_for_playback) / sample_rate
    rtf = inference_time / audio_duration
    print(f"Audio duration: {audio_duration:.3f} seconds")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"RTF: {rtf:.3f}") 