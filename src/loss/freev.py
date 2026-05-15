import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)


class FreeVLoss(nn.Module):
    def __init__(
        self,
        lambda_amplitude=45.0,
        lambda_phase=100.0,
        lambda_consistency=1.0,
        lambda_mel=45.0,
        lambda_adv=1.0,
        lambda_fm=2.0,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        sample_rate=22050,
        f_min=0.0,
        f_max=8000.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_amplitude = lambda_amplitude
        self.lambda_phase = lambda_phase
        self.lambda_consistency = lambda_consistency
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer("mel_basis", mel_basis.T)
        self.register_buffer("window", torch.hann_window(win_length))

    def _find_disc_names(self, model_output):
        names = set()
        for key in model_output:
            if key.endswith("_real") and key != "audio_real":
                name = key[:-5]
                if (
                    f"{name}_fake" in model_output
                    and f"{name}_fmap_real" in model_output
                    and f"{name}_fmap_fake" in model_output
                ):
                    names.add(name)
        return list(names)

    def forward(self, model_output, optimizer_idx):
        if optimizer_idx == 0:
            return self._generator_loss(model_output)
        else:
            return self._discriminator_loss(model_output)

    def _stft(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )

    def _compute_mel(self, audio):
        stft_out = self._stft(audio)
        magnitude = stft_out.abs()
        mel = torch.einsum("mf,bft->bmt", self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))

    def _align_time(self, *tensors):
        min_t = min(t.shape[-1] for t in tensors)
        return tuple(t[..., :min_t] for t in tensors)

    def _generator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        audio_real = model_output["audio_real"]
        audio_fake = model_output["audio_fake"]

        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        stft_real = self._stft(audio_real)
        stft_fake = self._stft(audio_fake)

        stft_real, stft_fake = self._align_time(stft_real, stft_fake)
        frames = stft_real.shape[-1]

        logamp_real = torch.log(stft_real.abs().clamp(min=1e-5))
        logamp_fake = torch.log(stft_fake.abs().clamp(min=1e-5))
        phase_real = torch.angle(stft_real)
        phase_fake = torch.angle(stft_fake)
        rea_real = stft_real.real
        rea_fake = stft_fake.real
        imag_real = stft_real.imag
        imag_fake = stft_fake.imag

        loss_amplitude = F.mse_loss(logamp_fake, logamp_real)
        losses["loss_amplitude"] = loss_amplitude

        loss_ip = torch.mean(anti_wrapping_function(phase_fake - phase_real))

        loss_gd = torch.mean(
            anti_wrapping_function(
                torch.diff(phase_fake, dim=-2) - torch.diff(phase_real, dim=-2)
            )
        )

        loss_ptd = torch.mean(
            anti_wrapping_function(
                torch.diff(phase_fake, dim=-1) - torch.diff(phase_real, dim=-1)
            )
        )

        loss_phase = loss_ip + loss_gd + loss_ptd
        losses["loss_phase"] = loss_phase
        losses["loss_phase_ip"] = loss_ip
        losses["loss_phase_gd"] = loss_gd
        losses["loss_phase_ptd"] = loss_ptd
        loss_consistency = torch.mean(
            torch.mean(
                (rea_fake - rea_real) ** 2 + (imag_fake - imag_real) ** 2,
                dim=(1, 2),
            )
        )
        losses["loss_consistency"] = loss_consistency

        mel_real = self._compute_mel(audio_real)
        mel_fake = self._compute_mel(audio_fake)
        loss_mel = F.l1_loss(mel_fake, mel_real)
        losses["loss_mel"] = loss_mel

        loss_adv = 0.0
        adv_count = 0
        for name in disc_names:
            for d_fake in model_output[f"{name}_fake"]:
                loss_adv += torch.mean(torch.clamp(1.0 - d_fake, min=0))
                adv_count += 1
        loss_adv = loss_adv / max(adv_count, 1)
        losses["loss_adv"] = loss_adv

        loss_fm = 0.0
        for name in disc_names:
            fmap_real_key = f"{name}_fmap_real"
            fmap_fake_key = f"{name}_fmap_fake"
            if fmap_real_key not in model_output:
                continue
            for real_fmaps, fake_fmaps in zip(
                model_output[fmap_real_key], model_output[fmap_fake_key]
            ):
                for real_f, fake_f in zip(real_fmaps, fake_fmaps):
                    real_f = real_f.detach()
                    slices = [slice(None)] * real_f.dim()
                    for d in range(2, real_f.dim()):
                        min_size = min(real_f.shape[d], fake_f.shape[d])
                        slices[d] = slice(None, min_size)
                    real_f = real_f[tuple(slices)]
                    fake_f = fake_f[tuple(slices)]
                    loss_fm += torch.mean(torch.abs(fake_f - real_f))
        losses["loss_fm"] = loss_fm

        losses["loss_g"] = (
            self.lambda_amplitude * loss_amplitude
            + self.lambda_phase * loss_phase
            + self.lambda_consistency * loss_consistency
            + self.lambda_mel * loss_mel
            + self.lambda_adv * loss_adv
            + self.lambda_fm * loss_fm
        )

        return losses

    def _discriminator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        loss_d = 0.0
        disc_count = 0
        for name in disc_names:
            for d_real, d_fake in zip(
                model_output[f"{name}_real"], model_output[f"{name}_fake"]
            ):
                loss_d += torch.mean(torch.clamp(1.0 - d_real, min=0))
                loss_d += torch.mean(torch.clamp(1.0 + d_fake.detach(), min=0))
                disc_count += 1
        loss_d = loss_d / max(disc_count, 1)
        losses["loss_d"] = loss_d

        return losses
