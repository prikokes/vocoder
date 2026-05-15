import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from torch.nn import functional as torch_functional


# transforms.py
class AudioToMelSpectrogram:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0):
        self.hop_length = hop_length
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode='reflect'
        )

    def __call__(self, audio):
        # audio: [C, T] или [T]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel = self.mel_transform(audio)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
