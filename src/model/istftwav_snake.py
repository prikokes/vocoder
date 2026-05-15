import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm
import math

from src.model.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from src.model.freev import ConvNeXtBlock, PseudoInverseMelFilter

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class Snake(nn.Module):
    def __init__(self, channels, is_2d=False):
        super().__init__()
        self.is_2d = is_2d
        shape = (1, channels, 1, 1) if is_2d else (1, channels, 1)
        self.alpha = nn.Parameter(torch.ones(*shape))

    def forward(self, x):
        a = self.alpha
        return x + (1.0 - torch.cos(2.0 * a * x)) / (2.0 * a + 1e-9)


class ResBlock1DWAV(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.acts1 = nn.ModuleList()
        self.acts2 = nn.ModuleList()
        for d in dilation:
            self.acts1.append(Snake(channels))
            self.convs1.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=get_padding(kernel_size, d)
            )))
            self.acts2.append(Snake(channels))
            self.convs2.append(weight_norm(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=1, padding=get_padding(kernel_size, 1)
            )))

    def forward(self, x):
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, self.acts1, self.acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ShuffleBlock2DWAV(nn.Module):
    def __init__(self, channels, n_layers=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            c_half = channels // 2
            self.blocks.append(nn.ModuleList([
                nn.Sequential(
                    weight_norm(nn.Conv2d(c_half, c_half, 3, padding=1, groups=c_half)),
                    weight_norm(nn.Conv2d(c_half, channels, 1))
                ),
                Snake(channels, is_2d=True),
                nn.Sequential(
                    weight_norm(nn.Conv2d(channels, channels, 3, padding=1, groups=channels)),
                    weight_norm(nn.Conv2d(channels, c_half, 1))
                ),
                Snake(c_half, is_2d=True)
            ]))

    @staticmethod
    def channel_shuffle(x, groups=2):
        B, C, H, W = x.shape
        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        return x.view(B, C, H, W)

    def forward(self, x):
        for conv1, act1, conv2, act2 in self.blocks:
            x1, x2 = x.chunk(2, dim=1)
            x2 = conv1(x2)
            x2 = act1(x2)
            x2 = conv2(x2)
            x2 = act2(x2)
            x = torch.cat([x1, x2], dim=1)
            x = self.channel_shuffle(x)
        return x

    def remove_weight_norm(self):
        for conv1, act1, conv2, act2 in self.blocks:
            remove_weight_norm(conv1[0])
            remove_weight_norm(conv1[1])
            remove_weight_norm(conv2[0])
            remove_weight_norm(conv2[1])


class ISTFTNet2WAV(nn.Module):
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
        n_2d_shuffle_layers=6,
        freq_upsample_factors=(2, 2, 2),
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.num_kernels = len(resblock_kernel_sizes)

        self.temporal_upsample = temporal_upsample
        self.sub_n_fft = n_fft // temporal_upsample
        self.sub_hop = hop_length // temporal_upsample
        self.sub_win = win_length // temporal_upsample
        self.sub_freq_bins = self.sub_n_fft // 2 + 1

        total_freq_up = int(np.prod(freq_upsample_factors))
        self.f_low = int(np.ceil(self.sub_freq_bins / total_freq_up))
        self.freq_upsample_factors = freq_upsample_factors

        ch_1d = channels // 2
        ch_concat = ch_1d * self.num_kernels
        ch_2d = channels // 4
        self.ch_2d = ch_2d

        self.conv_pre = weight_norm(nn.Conv1d(n_mels, channels, 7, padding=3))
        self.act_pre = Snake(channels)

        self.up = weight_norm(nn.ConvTranspose1d(
            channels, ch_1d, temporal_kernel_size,
            stride=temporal_upsample,
            padding=(temporal_kernel_size - temporal_upsample) // 2,
        ))

        self.resblocks_1d = nn.ModuleList()
        for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            self.resblocks_1d.append(ResBlock1DWAV(ch_1d, k, d))

        self.act_concat = Snake(ch_concat)
        self.reshape_proj = weight_norm(nn.Conv1d(ch_concat, ch_2d * self.f_low, 1))

        self.conv_2d_entry = weight_norm(nn.Conv2d(ch_2d, ch_2d, 3, padding=1))
        self.act_2d_entry = Snake(ch_2d, is_2d=True)

        self.shuffle_blocks = ShuffleBlock2DWAV(ch_2d, n_layers=n_2d_shuffle_layers)

        self.freq_ups = nn.ModuleList()
        self.freq_acts = nn.ModuleList()
        ch_cur = ch_2d
        for i, fu in enumerate(freq_upsample_factors):
            if i < len(freq_upsample_factors) - 1:
                ch_next = max(ch_cur // 2, 2)
            else:
                ch_next = 2
            self.freq_acts.append(Snake(ch_cur, is_2d=True))
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

        x = self.conv_pre(mel)
        x = self.act_pre(x)
        x = self.up(x)

        xs = []
        for resblock in self.resblocks_1d:
            xs.append(resblock(x))
        x = torch.cat(xs, dim=1)

        T_up = x.shape[-1]

        x = self.act_concat(x)
        x = self.reshape_proj(x)
        x = x.view(B, self.ch_2d, self.f_low, T_up)

        x = self.act_2d_entry(x)
        x = self.conv_2d_entry(x)
        x = self.shuffle_blocks(x)

        for act, freq_up in zip(self.freq_acts, self.freq_ups):
            x = act(x)
            x = freq_up(x)

        x = x[:, :, :self.sub_freq_bins, :]

        sub_mag = torch.exp(x[:, 0, :, :])
        sub_phase = torch.tanh(x[:, 1, :, :]) * math.pi

        sub_spec = torch.polar(sub_mag, sub_phase)

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

        return torch.atan2(full_stft.imag, full_stft.real)

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


class ISTFTWavSnake(nn.Module):
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
        phase_n_2d_shuffle_layers=6,
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

        self.phase_estimator = ISTFTNet2WAV(
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

        self.logamp = None
        self.phase = None
        self.spec_real = None
        self.spec_imag = None

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, mel, return_intermediates=True):
        inv_amp = self.pimf(mel)
        logamp = inv_amp.log()
        for conv_block in self.asp_convnext:
            logamp = conv_block(logamp, cond_embedding_id=None)

        phase = self.phase_estimator(mel)

        T = min(logamp.shape[-1], phase.shape[-1])
        logamp = logamp[..., :T]
        phase = phase[..., :T]

        amp = torch.exp(logamp)
        spec = torch.polar(amp, phase)

        audio = torch.istft(
            spec, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )

        if return_intermediates:
            self.logamp = logamp
            self.phase = phase
            self.spec_real = spec.real
            self.spec_imag = spec.imag
        else:
            self.logamp = None
            self.phase = None
            self.spec_real = None
            self.spec_imag = None

        return audio.unsqueeze(1)

    @torch.inference_mode()
    def inference_forward(self, mel):
        return self.forward(mel, return_intermediates=False)

    def remove_weight_norm(self):
        self.phase_estimator.remove_weight_norm()


class ISTFTWavSnakeGAN(nn.Module):
    def __init__(self, generator_params, mpd_params=None, mrd_params=None):
        super().__init__()
        self.generator = ISTFTWavSnake(**generator_params)
        self.mpd = MultiPeriodDiscriminator(**(mpd_params or {}))
        self.mrd = MultiResolutionDiscriminator(**(mrd_params or {}))
        self.discriminators = nn.ModuleList([self.mpd, self.mrd])

    def forward(self, mel=None, audio_real=None, **kwargs):
        if mel is None:
            if "mel" in kwargs:
                mel = kwargs["mel"]
            else:
                raise ValueError("Mel spectrogram input is required")

        audio_generated = self.generator(mel, return_intermediates=True)

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
        with torch.inference_mode():
            audio = self.generator(mel, return_intermediates=False)
        return {"audio_generated": audio}

    @property
    def discriminator(self):
        return self.discriminators