import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm
from librosa.filters import mel as librosa_mel_fn

import math

from src.model.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from src.model.freev import ConvNeXtBlock, PseudoInverseMelFilter


LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=get_padding(kernel_size, d)
            )))
            self.convs2.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=get_padding(kernel_size, 1)
            )))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ShuffleBlock2D(nn.Module):
    def __init__(self, channels, n_layers=3):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleList([
                weight_norm(nn.Conv2d(channels // 2, channels, 3, padding=1)),
                weight_norm(nn.Conv2d(channels, channels // 2, 3, padding=1)),
            ]))

    @staticmethod
    def channel_shuffle(x, groups=2):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        for conv1, conv2 in self.blocks:
            x1, x2 = x.chunk(2, dim=1)
            x2 = F.leaky_relu(x2, LRELU_SLOPE)
            x2 = conv1(x2)
            x2 = F.leaky_relu(x2, LRELU_SLOPE)
            x2 = conv2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.channel_shuffle(x)
        return x

    def remove_weight_norm(self):
        for conv1, conv2 in self.blocks:
            remove_weight_norm(conv1)
            remove_weight_norm(conv2)


class ISTFTNet2(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        channels=128,
        temporal_upsample=8,
        temporal_kernel_size=16,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        n_2d_shuffle_layers=3,
        freq_upsample_factors=(2, 2, 2),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_kernels = len(resblock_kernel_sizes)

        self.temporal_upsample = temporal_upsample
        self.sub_n_fft = n_fft // temporal_upsample        # 128
        self.sub_hop = hop_length // temporal_upsample     # 32
        self.sub_win = win_length // temporal_upsample     # 128
        self.sub_freq_bins = self.sub_n_fft // 2 + 1       # 65

        total_freq_up = int(np.prod(freq_upsample_factors)) # 8
        self.f_low = int(np.ceil(self.sub_freq_bins / total_freq_up))  # ceil(65/8) = 9
        self.freq_upsample_factors = freq_upsample_factors

        ch_1d = channels // 2                 # 64
        ch_concat = ch_1d * self.num_kernels  # 192
        ch_2d = channels // 4                 # 32
        self.ch_2d = ch_2d

        self.conv_pre = weight_norm(nn.Conv1d(n_mels, channels, 7, padding=3))

        self.up = weight_norm(nn.ConvTranspose1d(
            channels, ch_1d, temporal_kernel_size,
            stride=temporal_upsample,
            padding=(temporal_kernel_size - temporal_upsample) // 2,
        ))

        self.resblocks_1d = nn.ModuleList()
        for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.resblocks_1d.append(ResBlock1D(ch_1d, k, d))

        self.reshape_proj = weight_norm(
            nn.Conv1d(ch_concat, ch_2d * self.f_low, 1)
        )

        # Layout: (B, C, freq, time)
        self.conv_2d_entry = weight_norm(nn.Conv2d(ch_2d, ch_2d, 3, padding=1))
        self.shuffle_blocks = ShuffleBlock2D(ch_2d, n_layers=n_2d_shuffle_layers)

        self.freq_ups = nn.ModuleList()
        ch_cur = ch_2d
        for i, fu in enumerate(freq_upsample_factors):
            if i < len(freq_upsample_factors) - 1:
                ch_next = max(ch_cur // 2, 2)
            else:
                ch_next = 2  # mag + phase
            self.freq_ups.append(weight_norm(nn.ConvTranspose2d(
                ch_cur, ch_next,
                kernel_size=(fu * 2, 3),
                stride=(fu, 1),
                padding=(fu // 2, 1),
            )))
            ch_cur = ch_next

        self.register_buffer("sub_window", torch.hann_window(self.sub_win))
        self.register_buffer("full_window", torch.hann_window(win_length))

    def forward(self, mel):
        B = mel.shape[0]

        x = self.conv_pre(mel)                         # (B, C, T)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.up(x)                                 # (B, C/2, 8T)

        xs = []
        for resblock in self.resblocks_1d:
            xs.append(resblock(x))
        x = torch.cat(xs, dim=1)                       # (B, C/2*3, 8T)

        T_up = x.shape[-1]

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.reshape_proj(x)                       # (B, ch_2d*f_low, 8T)
        x = x.view(B, self.ch_2d, self.f_low, T_up)    # (B, ch_2d, freq=f_low, time=8T)

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_2d_entry(x)                      # (B, ch_2d, f_low, 8T)
        x = self.shuffle_blocks(x)                     # (B, ch_2d, f_low, 8T)

        for freq_up in self.freq_ups:
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = freq_up(x)
        # (B, 2, f_low*8, 8T) = (B, 2, 72, 8T)

        x = x[:, :, :self.sub_freq_bins, :]            # (B, 2, 65, 8T)

        sub_mag = torch.exp(x[:, 0, :, :])             # (B, 65, 8T)
        sub_phase = torch.tanh(x[:, 1, :, :]) * math.pi

        sub_spec = sub_mag * torch.exp(1j * sub_phase)  # (B, 65, 8T)

        intermediate_wav = torch.istft(
            sub_spec,
            n_fft=self.sub_n_fft,
            hop_length=self.sub_hop,
            win_length=self.sub_win,
            window=self.sub_window,
        )

        full_stft = torch.stft(
            intermediate_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.full_window,
            center=True,
            return_complex=True,
        )

        return torch.angle(full_stft)

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.up)
        remove_weight_norm(self.reshape_proj)
        remove_weight_norm(self.conv_2d_entry)
        self.shuffle_blocks.remove_weight_norm()
        for fu in self.freq_ups:
            remove_weight_norm(fu)
        for block in self.resblocks_1d:
            block.remove_weight_norm()


class ISTFTWav(nn.Module):
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
        asp_intermediate_dim=1536,
        asp_num_layers_for_scale=8,
        phase_channels=128,
        phase_temporal_upsample=8,
        phase_temporal_kernel_size=16,
        phase_resblock_kernel_sizes=(3, 7, 11),
        phase_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        phase_n_2d_shuffle_layers=3,
        phase_freq_upsample_factors=(2, 2, 2),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_bins = n_fft // 2 + 1

        self.pimf = PseudoInverseMelFilter(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

        layer_scale = 1.0 / asp_num_layers_for_scale
        self.asp_convnext = nn.ModuleList([
            ConvNeXtBlock(
                dim=asp_channel,
                intermediate_dim=asp_intermediate_dim,
                layer_scale_init_value=layer_scale,
            )
            for _ in range(asp_num_convnext_blocks)
        ])

        self.phase_estimator = ISTFTNet2(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            channels=phase_channels,
            temporal_upsample=phase_temporal_upsample,
            temporal_kernel_size=phase_temporal_kernel_size,
            resblock_kernel_sizes=phase_resblock_kernel_sizes,
            resblock_dilation_sizes=phase_resblock_dilation_sizes,
            n_2d_shuffle_layers=phase_n_2d_shuffle_layers,
            freq_upsample_factors=phase_freq_upsample_factors,
        )

        self.register_buffer("window", torch.hann_window(win_length))
        self.asp_convnext.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, mel):
        inv_amp = self.pimf(mel)
        logamp = inv_amp.log()
        for conv_block in self.asp_convnext:
            logamp = conv_block(logamp, cond_embedding_id=None)

        phase = self.phase_estimator(mel)

        T = min(logamp.shape[-1], phase.shape[-1])
        logamp = logamp[..., :T]
        phase = phase[..., :T]

        rea = torch.exp(logamp) * torch.cos(phase)
        imag = torch.exp(logamp) * torch.sin(phase)
        spec = torch.complex(rea, imag)

        audio = torch.istft(
            spec, self.n_fft,
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

    def remove_weight_norm(self):
        for block in self.asp_convnext:
            if hasattr(block, "remove_weight_norm"):
                block.remove_weight_norm()


class ISTFTWavGAN(nn.Module):
    def __init__(self, generator_params, mpd_params=None, mrd_params=None):
        super().__init__()
        self.generator = ISTFTWav(**generator_params)
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