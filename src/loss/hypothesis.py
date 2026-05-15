import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


def anti_wrapping_function(x):
    return torch.abs(x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi)


class HypothesisLoss(nn.Module):
    def __init__(
        self,
        lambda_amplitude=45.0,
        lambda_phase_ip=100.0,
        lambda_phase_gd=100.0,
        lambda_phase_ptd=100.0,
        lambda_consistency=20.0,
        lambda_real_imag=0.0,
        lambda_mel=45.0,
        lambda_adv=1.0,
        lambda_fm=1.0, 
        mrd_weight=1.0,
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
        self.lambda_phase_ip = lambda_phase_ip
        self.lambda_phase_gd = lambda_phase_gd
        self.lambda_phase_ptd = lambda_phase_ptd
        self.lambda_consistency = lambda_consistency
        self.lambda_real_imag = lambda_real_imag
        self.lambda_mel = lambda_mel
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.mrd_weight = mrd_weight

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

    def _get_disc_weight(self, name):
        if "mrd" in name.lower() or "resolution" in name.lower():
            return self.mrd_weight
        return 1.0

    def _find_disc_names(self, model_output):
        names = set()
        for key in model_output:
            if key.endswith("_real") and key not in ("audio_real",):
                name = key[:-5]
                if (
                    f"{name}_fake" in model_output
                    and f"{name}_fmap_real" in model_output
                    and f"{name}_fmap_fake" in model_output
                ):
                    names.add(name)
        return list(names)

    def _compute_mel_from_audio(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        stft_out = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        mag = stft_out.abs()
        mel = torch.einsum("mf,bft->bmt", self.mel_basis, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel

    def _compute_gt_spectral(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        stft_out = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
        )
        log_mag = torch.log(stft_out.abs().clamp(min=1e-5))
        phase = torch.angle(stft_out)
        rea = stft_out.real
        imag = stft_out.imag
        return log_mag, phase, rea, imag

    def forward(self, model_output, optimizer_idx):
        if optimizer_idx == 0:
            return self._generator_loss(model_output)
        else:
            return self._discriminator_loss(model_output)

    def _generator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        audio_real = model_output["audio_real"]
        audio_fake = model_output["audio_fake"]

        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        logamp_fake = model_output["logamp"]
        phase_fake = model_output["phase"]
        rea_fake = model_output["spec_real"]
        imag_fake = model_output["spec_imag"]

        logamp_real, phase_real, rea_real, imag_real = self._compute_gt_spectral(
            audio_real
        )

        T = min(logamp_fake.shape[-1], logamp_real.shape[-1])
        logamp_fake = logamp_fake[..., :T]
        phase_fake = phase_fake[..., :T]
        rea_fake = rea_fake[..., :T]
        imag_fake = imag_fake[..., :T]
        logamp_real = logamp_real[..., :T]
        phase_real = phase_real[..., :T]
        rea_real = rea_real[..., :T]
        imag_real = imag_real[..., :T]

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

        loss_phase = (
            self.lambda_phase_ip * loss_ip
            + self.lambda_phase_gd * loss_gd
            + self.lambda_phase_ptd * loss_ptd
        )
        losses["loss_phase"] = loss_phase
        losses["loss_phase_ip"] = loss_ip
        losses["loss_phase_gd"] = loss_gd
        losses["loss_phase_ptd"] = loss_ptd

        _, _, rea_from_audio, imag_from_audio = self._compute_gt_spectral(audio_fake)
        rea_from_audio = rea_from_audio[..., :T]
        imag_from_audio = imag_from_audio[..., :T]

        loss_consistency = torch.mean(
            (rea_fake - rea_from_audio) ** 2 + (imag_fake - imag_from_audio) ** 2
        )
        losses["loss_consistency"] = loss_consistency

        loss_real = F.l1_loss(rea_fake, rea_real)
        loss_imag = F.l1_loss(imag_fake, imag_real)
        losses["loss_real"] = loss_real
        losses["loss_imag"] = loss_imag

        mel_real = self._compute_mel_from_audio(audio_real)
        mel_fake = self._compute_mel_from_audio(audio_fake)
        loss_mel = F.l1_loss(mel_fake, mel_real)
        losses["loss_mel"] = loss_mel

        loss_adv = 0.0
        for name in disc_names:
            w = self._get_disc_weight(name)
            
            adv_loss_part = 0.0  
            
            for d_fake in model_output[f"{name}_fake"]:
                adv_loss_part += torch.mean(torch.clamp(1.0 - d_fake, min=0.0))
                
            loss_adv += w * adv_loss_part
            
        losses["loss_adv"] = loss_adv
        
        loss_fm = 0.0
        for name in disc_names:
            w = self._get_disc_weight(name)
            fm_loss = 0.0
            fmap_real_key = f"{name}_fmap_real"
            fmap_fake_key = f"{name}_fmap_fake"
            if fmap_real_key not in model_output:
                continue
            for real_fmaps, fake_fmaps in zip(
                model_output[fmap_real_key], model_output[fmap_fake_key]
            ):
                for real_f, fake_f in zip(real_fmaps, fake_fmaps):
                    real_f = real_f.detach()
                    if real_f.shape != fake_f.shape:
                        min_t = min(real_f.shape[-1], fake_f.shape[-1])
                        real_f = real_f[..., :min_t]
                        fake_f = fake_f[..., :min_t]
                    fm_loss += torch.mean(torch.abs(fake_f - real_f))
            loss_fm += w * fm_loss
        losses["loss_fm"] = loss_fm

        losses["loss_g"] = (
            self.lambda_amplitude * loss_amplitude
            + loss_phase                                     
            + self.lambda_consistency * loss_consistency      
            + self.lambda_real_imag * (loss_real + loss_imag)
            + self.lambda_mel * loss_mel                     
            + self.lambda_adv * loss_adv                     
            + self.lambda_fm * loss_fm                       
        )

        return losses

    def _discriminator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        loss_d = 0.0
        for name in disc_names:
            w = self._get_disc_weight(name)
            disc_loss = 0.0
            for d_real, d_fake in zip(
                model_output[f"{name}_real"], model_output[f"{name}_fake"]
            ):
                disc_loss += torch.mean(torch.clamp(1.0 - d_real, min=0.0))
                disc_loss += torch.mean(torch.clamp(1.0 + d_fake.detach(), min=0.0))
            loss_d += w * disc_loss
        losses["loss_d"] = loss_d

        return losses