import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from src.model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator

LRELU_SLOPE = 0.1


class HiFiGAN(nn.Module):
    def __init__(self, config):
        super(HiFiGAN, self).__init__()

        class HParams:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)

        h = HParams(
            resblock_kernel_sizes=config.get('resblock_kernel_sizes', [3, 7, 11]),
            resblock_dilation_sizes=config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
            upsample_rates=config.get('upsample_rates', [8, 8, 2, 2]),
            upsample_kernel_sizes=config.get('upsample_kernel_sizes', [16, 16, 4, 4]),
            upsample_initial_channel=config.get('upsample_initial_channel', 512),
            num_mels=config.get('num_mels', 80),
            resblock=config.get('resblock_type', '1')
        )

        self.generator = Generator(h)
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        self.discriminators = nn.ModuleList([self.mpd, self.msd])

        self.config = config

    def forward(self, mel=None, audio_real=None, **kwargs):
        if mel is None:
            if 'mel' in kwargs:
                mel = kwargs['mel']
            elif 'data_object' in kwargs:
                mel = kwargs['data_object']
            else:
                raise ValueError("Mel spectrogram input is required")

        audio_generated = self.generator(mel)

        if audio_real is not None:
            mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.mpd(audio_real, audio_generated)
            msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = self.msd(audio_real, audio_generated)

            return {
                'audio_generated': audio_generated,
                'mpd_real': mpd_real,
                'mpd_fake': mpd_fake,
                'msd_real': msd_real,
                'msd_fake': msd_fake,
                'mpd_real_fmaps': mpd_real_fmaps,
                'mpd_fake_fmaps': mpd_fake_fmaps,
                'msd_real_fmaps': msd_real_fmaps,
                'msd_fake_fmaps': msd_fake_fmaps
            }
        else:
            return {'audio_generated': audio_generated}

    def inference(self, mel):
        return self.forward(mel=mel)

    def remove_weight_norm(self):
        self.generator.remove_weight_norm()

    @property
    def discriminator(self):
        """Combine MPD and MSD for parameter access"""
        return self.discriminators


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
