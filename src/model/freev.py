import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from librosa.filters import mel as librosa_mel_fn

from src.model.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class PseudoInverseMelFilter(nn.Module):
    def __init__(self, sr=22050, n_fft=1024, n_mels=80, fmin=0.0, fmax=8000.0):
        super().__init__()
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        self.register_buffer("mel_basis_pinv", torch.FloatTensor(mel_basis_pinv))

    def forward(self, log_mel, return_log=False):
        mel_linear = torch.exp(log_mel)
        magnitude = torch.matmul(self.mel_basis_pinv, mel_linear)
        magnitude.clamp_(min=1e-5)
        if return_log:
            return magnitude.log_()
        return magnitude


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x



class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale_init_value=None,
                 adanorm_num_embeddings=None):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.grn = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x, cond_embedding_id=None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose(1, 2)
        return residual + x


class FreeVGenerator(nn.Module):
    def __init__(
        self,
        sr=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        fmin=0.0,
        fmax=8000.0,
        asp_channel=513,
        asp_num_convnext_blocks=1,
        psp_channel=512,
        psp_input_conv_kernel_size=7,
        psp_output_R_conv_kernel_size=7,
        psp_output_I_conv_kernel_size=7,
        psp_num_convnext_blocks=8,
        intermediate_dim=1536,
        num_layers_for_scale=8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_bins = n_fft // 2 + 1  # 513 for n_fft=1024

        self.pimf = PseudoInverseMelFilter(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

        layer_scale_init_value = 1.0 / num_layers_for_scale

        self.asp_convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=asp_channel,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
            )
            for _ in range(asp_num_convnext_blocks)
        ])

        self.psp_input_conv = nn.Conv1d(
            n_mels,
            psp_channel,
            psp_input_conv_kernel_size,
            stride=1,
            padding=get_padding(psp_input_conv_kernel_size, 1),
        )
        self.psp_input_norm = nn.LayerNorm(psp_channel, eps=1e-6)

        self.psp_convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=psp_channel,
                intermediate_dim=intermediate_dim,
                layer_scale_init_value=layer_scale_init_value,
            )
            for _ in range(psp_num_convnext_blocks)
        ])

        self.psp_output_norm = nn.LayerNorm(psp_channel, eps=1e-6)

        self.psp_output_R_conv = nn.Conv1d(
            psp_channel,
            self.freq_bins,
            psp_output_R_conv_kernel_size,
            stride=1,
            padding=get_padding(psp_output_R_conv_kernel_size, 1),
        )
        self.psp_output_I_conv = nn.Conv1d(
            psp_channel,
            self.freq_bins,
            psp_output_I_conv_kernel_size,
            stride=1,
            padding=get_padding(psp_output_I_conv_kernel_size, 1),
        )

        self.register_buffer("window", torch.hann_window(win_length))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, mel):
        inv_amp = self.pimf(mel)  # (B, freq_bins, T)

        logamp = inv_amp.log()  # (B, freq_bins, T)
        for conv_block in self.asp_convnext:
            logamp = conv_block(logamp, cond_embedding_id=None)

        pha = self.psp_input_conv(mel)          # (B, psp_channel, T)
        pha = self.psp_input_norm(pha.transpose(1, 2)).transpose(1, 2)

        for conv_block in self.psp_convnext:
            pha = conv_block(pha, cond_embedding_id=None)

        pha = self.psp_output_norm(pha.transpose(1, 2)).transpose(1, 2)

        R = self.psp_output_R_conv(pha)         # (B, freq_bins, T)
        I = self.psp_output_I_conv(pha)         # (B, freq_bins, T)
        phase = torch.atan2(I, R)               # (B, freq_bins, T)

        rea = torch.exp(logamp) * torch.cos(phase)
        imag = torch.exp(logamp) * torch.sin(phase)
        spec = torch.complex(rea, imag)         # (B, freq_bins, T)

        audio = torch.istft(
            spec,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )

        self.logamp = logamp
        self.phase = phase
        self.spec_real = rea
        self.spec_imag = imag

        return audio.unsqueeze(1)  


class FreeV(nn.Module):
    def __init__(self, generator_params, mpd_params=None, mrd_params=None):
        super().__init__()
        self.generator = FreeVGenerator(**generator_params)
        self.mpd = MultiPeriodDiscriminator(**(mpd_params or {}))
        self.mrd = MultiResolutionDiscriminator(**(mrd_params or {}))
        self.discriminators = nn.ModuleList([self.mpd, self.mrd])

    def forward(self, mel=None, audio_real=None, **kwargs):
        if mel is None:
            if "mel" in kwargs:
                mel = kwargs["mel"]
            else:
                raise ValueError("Mel spectrogram input is required")

        audio_generated = self.generator(mel)

        result = {
            "audio_generated": audio_generated,
            "logamp": self.generator.logamp,
            "phase": self.generator.phase,
            "spec_real": self.generator.spec_real,
            "spec_imag": self.generator.spec_imag,
        }

        if audio_real is not None:
            min_len = min(audio_real.shape[-1], audio_generated.shape[-1])
            audio_real_t = audio_real[..., :min_len]
            audio_gen_t = audio_generated[..., :min_len]

            mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.mpd(
                audio_real_t, audio_gen_t
            )
            mrd_real, mrd_fake, mrd_real_fmaps, mrd_fake_fmaps = self.mrd(
                audio_real_t, audio_gen_t
            )

            result.update({
                "mpd_real": mpd_real,
                "mpd_fake": mpd_fake,
                "mrd_real": mrd_real,
                "mrd_fake": mrd_fake,
                "mpd_real_fmaps": mpd_real_fmaps,
                "mpd_fake_fmaps": mpd_fake_fmaps,
                "mrd_real_fmaps": mrd_real_fmaps,
                "mrd_fake_fmaps": mrd_fake_fmaps,
            })

        return result

    def inference(self, mel):
        self.generator.eval()
        with torch.no_grad():
            audio = self.generator(mel)
        return {"audio_generated": audio}

    @property
    def discriminator(self):
        return self.discriminators
