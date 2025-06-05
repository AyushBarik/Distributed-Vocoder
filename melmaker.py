import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchaudio.transforms as TT

wav_file_path = 'example.wav'
output_data_path = 'mel_spectrogram_data.npy'
output_image_path = 'mel_spectrogram_image.png'

# Load audio with 22050 Hz sample rate (DiffWave's expected rate)
y, sr = librosa.load(wav_file_path, sr=22050)

# DiffWave's exact preprocessing parameters
mel_args = {
    'sample_rate': sr,
    'win_length': 256 * 4,  # hop_samples * 4
    'hop_length': 256,      # hop_samples 
    'n_fft': 1024,
    'f_min': 20.0,
    'f_max': sr / 2.0,
    'n_mels': 80,
    'power': 1.0,           # Use power=1.0 for linear scale
    'normalized': True,
}

# Use PyTorch's mel spectrogram transform (same as DiffWave uses)
mel_spec_transform = TT.MelSpectrogram(**mel_args)
audio_tensor = torch.from_numpy(y)

# Generate mel spectrogram in DiffWave's exact format
with torch.no_grad():
    S_mel = mel_spec_transform(audio_tensor)
    # Apply DiffWave's exact normalization
    S_mel_processed = 20 * torch.log10(torch.clamp(S_mel, min=1e-5)) - 20
    S_mel_normalized = torch.clamp((S_mel_processed + 100) / 100, 0.0, 1.0)

# Save the properly processed matrix for DiffWave
np.save(output_data_path, S_mel_normalized.numpy())
print(f"DiffWave-compatible mel spectrogram saved to: {output_data_path}")
print(f"Shape: {S_mel_normalized.shape}")
print(f"Value range: {S_mel_normalized.min():.3f} to {S_mel_normalized.max():.3f}")

# Create visualization using the ACTUAL saved normalized data
plt.figure(figsize=(12, 4))
librosa.display.specshow(S_mel_normalized.numpy(), sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='viridis')
plt.colorbar(format='%.3f')
plt.title('Mel Spectrogram (Normalized 0-1 - Actual Saved Data)')
plt.tight_layout()
plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Mel spectrogram image saved to: {output_image_path}")