import torch
import torch.nn.functional as F
import numpy as np
from torchaudio.functional import melscale_fbanks
from pystoi import stoi
from pesq import pesq


class MelDistance:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0, name="MelDistance"):
        self.name = name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate
        ).T  # [n_mels, n_freqs]

    def __call__(self, audio_real, audio_fake):
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        mel_real = self._compute_mel(audio_real)
        mel_fake = self._compute_mel(audio_fake)

        return F.l1_loss(mel_fake, mel_real).item()

    def _compute_mel(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True
        )
        magnitude = torch.abs(stft)
        mel_basis = self.mel_basis.to(audio.device)
        mel = torch.einsum('mf,bft->bmt', mel_basis, magnitude)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel


class LSD:
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, name="LSD"):
        self.name = name
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __call__(self, audio_real, audio_fake):
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        if audio_real.dim() == 3:
            audio_real = audio_real.squeeze(1)
        if audio_fake.dim() == 3:
            audio_fake = audio_fake.squeeze(1)

        window = torch.hann_window(self.win_length, device=audio_real.device)

        stft_real = torch.stft(
            audio_real, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True
        )
        stft_fake = torch.stft(
            audio_fake, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True
        )

        power_real = torch.abs(stft_real) ** 2
        power_fake = torch.abs(stft_fake) ** 2

        log_power_real = torch.log10(torch.clamp(power_real, min=1e-10))
        log_power_fake = torch.log10(torch.clamp(power_fake, min=1e-10))

        lsd = torch.sqrt(torch.mean((log_power_real - log_power_fake) ** 2, dim=1))
        return torch.mean(lsd).item()


class PESQ:
    def __init__(self, sample_rate=22050, name="PESQ"):
        self.name = name
        self.sample_rate = sample_rate

    def __call__(self, audio_real, audio_fake):
        real = audio_real[0].squeeze().detach().cpu().numpy()
        fake = audio_fake[0].squeeze().detach().cpu().numpy()

        min_len = min(len(real), len(fake))
        real = real[:min_len]
        fake = fake[:min_len]

        if self.sample_rate == 16000:
            score = pesq(16000, real, fake, 'wb')
        elif self.sample_rate == 8000:
            score = pesq(8000, real, fake, 'nb')
        else:
            import librosa
            real_16k = librosa.resample(real, orig_sr=self.sample_rate, target_sr=16000)
            fake_16k = librosa.resample(fake, orig_sr=self.sample_rate, target_sr=16000)
            score = pesq(16000, real_16k, fake_16k, 'wb')
        return float(score)



class STOI:
    def __init__(self, sample_rate=22050, name="STOI"):
        self.name = name
        self.sample_rate = sample_rate

    def __call__(self, audio_real, audio_fake):
        from pystoi import stoi

        real = audio_real[0].squeeze().detach().cpu().numpy().astype('float64')
        fake = audio_fake[0].squeeze().detach().cpu().numpy().astype('float64')

        print(f"STOI debug: real shape={real.shape}, fake shape={fake.shape}")

        min_len = min(len(real), len(fake))
        real = real[:min_len]
        fake = fake[:min_len]

        print(f"STOI debug after trim: len={min_len}, "
              f"duration={min_len / self.sample_rate:.3f}s, "
              f"real range=[{real.min():.4f}, {real.max():.4f}], "
              f"fake range=[{fake.min():.4f}, {fake.max():.4f}]")

        if min_len < self.sample_rate * 0.5:
            print(f"STOI: too short ({min_len / self.sample_rate:.3f}s < 0.5s)")
            return 0.0

        try:
            score = stoi(real, fake, self.sample_rate, extended=False)
            print(f"STOI score: {score}")
            return float(score)
        except Exception as e:
            print(f"STOI error: {e}")
            return 0.0
