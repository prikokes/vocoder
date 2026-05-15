import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_fm=2.0, lambda_mel=45.0,
                 n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=80, sample_rate=22050, f_min=0.0, f_max=8000.0):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer('mel_basis', mel_basis.T)

    def _find_disc_names(self, model_output):
        names = set()
        for key in model_output:
            if key.endswith("_real") and key != "audio_real":
                name = key[:-5]
                if (f"{name}_fake" in model_output
                        and f"{name}_fmap_real" in model_output
                        and f"{name}_fmap_fake" in model_output):
                    names.add(name)
        return list(names)

    def forward(self, model_output, optimizer_idx):
        if optimizer_idx == 0:
            return self._generator_loss(model_output)
        else:
            return self._discriminator_loss(model_output)

    def _generator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        adv_loss = 0.0
        adv_count = 0
        for name in disc_names:
            for d_fake in model_output[f"{name}_fake"]:
                adv_loss += self.mse_loss(d_fake, torch.ones_like(d_fake))
                adv_count += 1
        adv_loss = adv_loss / max(adv_count, 1)
        losses['loss_adv'] = adv_loss

        fm_loss = 0.0
        fm_count = 0
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

                    fm_loss += self.l1_loss(fake_f, real_f)
                    fm_count += 1

        fm_loss = fm_loss / max(fm_count, 1)
        losses['loss_fm'] = fm_loss

        mel_loss = self._mel_spectrogram_loss(
            model_output["audio_real"], model_output["audio_fake"]
        )
        losses['loss_mel'] = mel_loss

        losses['loss_g'] = (
            self.lambda_adv * adv_loss +
            self.lambda_fm * fm_loss +
            self.lambda_mel * mel_loss
        )
        return losses

    def _discriminator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        disc_loss = 0.0
        disc_count = 0
        for name in disc_names:
            for d_real, d_fake in zip(
                model_output[f"{name}_real"], model_output[f"{name}_fake"]
            ):
                disc_loss += self.mse_loss(d_real, torch.ones_like(d_real))
                disc_loss += self.mse_loss(d_fake.detach(), torch.zeros_like(d_fake))
                disc_count += 1

        disc_loss = disc_loss / max(2 * disc_count, 1)
        losses['loss_d'] = disc_loss
        return losses

    def _mel_spectrogram_loss(self, audio_real, audio_fake):
        min_length = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_length]
        audio_fake = audio_fake[..., :min_length]
        return self.l1_loss(self._compute_mel(audio_fake), self._compute_mel(audio_real))

    def _compute_mel(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True,
        )
        magnitude = torch.abs(stft)
        mel = torch.einsum('mf,bft->bmt', self.mel_basis.to(audio.device), magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))