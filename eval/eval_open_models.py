import os
import sys
import json
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from librosa.filters import mel as librosa_mel_fn


mel_basis = {}
hann_window = {}

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def mel_spectrogram_hifigan(y, n_fft=1024, num_mels=80, sampling_rate=22050,
                             hop_size=256, win_size=1024, fmin=0, fmax=8000,
                             center=False):
    global mel_basis, hann_window

    if y.dim() == 1:
        y = y.unsqueeze(0)

    key = f"{fmax}_{num_mels}_{sampling_rate}_{y.device}"
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    win_key = f"{win_size}_{y.device}"
    if win_key not in hann_window:
        hann_window[win_key] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=hann_window[win_key],
                      center=center, pad_mode='reflect',
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)

    spec = torch.matmul(mel_basis[key], spec)
    spec = dynamic_range_compression_torch(spec)
    return spec


LJSPEECH_ROOT     = "E:/se/4_course/audio_vocoder/data/LJSpeech-1.1/LJSpeech-1.1"
LJSPEECH_WAVS_DIR = LJSPEECH_ROOT + "/wavs"
METADATA_FILE     = LJSPEECH_ROOT + "/metadata.csv"
SPLIT_SYMBOL      = "|"

OFFSET       = 12500
LIMIT        = 600
SEGMENT_SIZE = 16384
USE_SEGMENT  = True
SEGMENT_MODE = "first"

HIFIGAN_V1_CKPT   = "E:/se/4_course/thesis/eval_open_models/checkpoints/generator_v1"
HIFIGAN_V1_CONFIG = "E:/se/4_course/thesis/eval_open_models/configs/config_v1.json"
HIFIGAN_V2_CKPT   = "E:/se/4_course/thesis/eval_open_models/checkpoints/generator_v2"
HIFIGAN_V2_CONFIG = "E:/se/4_course/thesis/eval_open_models/configs/config_v2.json"
HIFIGAN_V3_CKPT   = "E:/se/4_course/thesis/eval_open_models/checkpoints/generator_v3"
HIFIGAN_V3_CONFIG = "E:/se/4_course/thesis/eval_open_models/configs/config_v3.json"
HIFIGAN_REPO_PATH = "E:/se/4_course/thesis/eval_open_models/hifi-gan"

FREEV_CKPT      = "E:/se/4_course/thesis/eval_open_models/checkpoints/freev_g_01000000"
FREEV_CONFIG    = "E:/se/4_course/thesis/eval_open_models/FreeV/config2.json"
FREEV_REPO_PATH = "E:/se/4_course/thesis/eval_open_models/FreeV"

MODELS_TO_EVAL = ["hifigan_v1", "hifigan_v2", "hifigan_v3", "vocos", "freev"]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON    = "benchmark_results.json"


def pesq_metric(y_pred, y_true, sr):
    from pesq import pesq
    target_sr = 16000
    if sr != target_sr:
        y_pred = torchaudio.functional.resample(y_pred, sr, target_sr)
        y_true = torchaudio.functional.resample(y_true, sr, target_sr)
    y_pred_np = y_pred.squeeze().cpu().numpy().astype(np.float32)
    y_true_np = y_true.squeeze().cpu().numpy().astype(np.float32)
    try:
        return pesq(target_sr, y_true_np, y_pred_np, 'wb')
    except Exception:
        return float('nan')


def mel_distance_metric(y_pred, y_true, sr):
    if sr != 22050:
        y_pred = torchaudio.functional.resample(y_pred, sr, 22050)
        y_true = torchaudio.functional.resample(y_true, sr, 22050)
    yp = y_pred.squeeze(0) if y_pred.dim() > 1 else y_pred
    yt = y_true.squeeze(0) if y_true.dim() > 1 else y_true
    mel_pred = mel_spectrogram_hifigan(yp)
    mel_true = mel_spectrogram_hifigan(yt)
    min_t = min(mel_pred.shape[-1], mel_true.shape[-1])
    return F.l1_loss(mel_pred[..., :min_t], mel_true[..., :min_t]).item()

def stoi_metric(y_pred, y_true, sr):
    from pystoi import stoi
    target_sr = 16000
    if sr != target_sr:
        y_pred = torchaudio.functional.resample(y_pred, sr, target_sr)
        y_true = torchaudio.functional.resample(y_true, sr, target_sr)
    y_pred_np = y_pred.squeeze().cpu().numpy().astype(np.float32)
    y_true_np = y_true.squeeze().cpu().numpy().astype(np.float32)
    try:
        return stoi(y_true_np, y_pred_np, target_sr, extended=False)
    except Exception:
        return float('nan')


def lsd_metric(y_pred, y_true, sr, n_fft=1024, hop_length=256, target_sr=22050):
    if sr != target_sr:
        y_pred = torchaudio.functional.resample(y_pred, sr, target_sr)
        y_true = torchaudio.functional.resample(y_true, sr, target_sr)
    spec_pred = torch.stft(y_pred.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                           window=torch.hann_window(n_fft).to(y_pred.device),
                           return_complex=True).abs()
    spec_true = torch.stft(y_true.squeeze(0), n_fft=n_fft, hop_length=hop_length,
                           window=torch.hann_window(n_fft).to(y_true.device),
                           return_complex=True).abs()
    log_pred = torch.log10(spec_pred.clamp(min=1e-8) ** 2)
    log_true = torch.log10(spec_true.clamp(min=1e-8) ** 2)
    diff = (log_pred - log_true) ** 2
    return torch.sqrt(diff.mean(dim=0)).mean().item()


def compute_all_metrics(y_pred, y_true, sr):
    min_len = min(y_pred.shape[-1], y_true.shape[-1])
    y_pred = y_pred[..., :min_len]
    y_true = y_true[..., :min_len]
    return {
        "lsd":          lsd_metric(y_pred, y_true, sr),
        "pesq":         pesq_metric(y_pred, y_true, sr),
        "stoi":         stoi_metric(y_pred, y_true, sr),
        "mel_distance": mel_distance_metric(y_pred, y_true, sr),
    }


class HiFiGANWrapper:
    def __init__(self, ckpt_path, config_path, device):
        if HIFIGAN_REPO_PATH not in sys.path:
            sys.path.insert(0, HIFIGAN_REPO_PATH)
        from models import Generator
        from env import AttrDict

        with open(config_path) as f:
            h = AttrDict(json.load(f))
        self.h = h
        self.generator = Generator(h).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.generator.load_state_dict(state_dict['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.device = device
        self.sr = h.sampling_rate
        n_params = sum(p.numel() for p in self.generator.parameters())
        print(f"  HiFi-GAN params: {n_params/1e6:.2f}M, sr={self.sr}")

    @torch.no_grad()
    def __call__(self, audio_gt):
        h = self.h
        mel = mel_spectrogram_hifigan(
            audio_gt,
            n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate=h.sampling_rate,
            hop_size=h.hop_size, win_size=h.win_size,
            fmin=h.fmin, fmax=h.fmax,
        )
        audio = self.generator(mel).squeeze(1)
        return audio


class VocosWrapper:
    def __init__(self, device):
        from vocos import Vocos
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        self.model.eval()
        self.device = device
        self.sr = 24000
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  Vocos params: {n_params/1e6:.2f}M, sr={self.sr}")

    @torch.no_grad()
    def __call__(self, audio_gt):
        features = self.model.feature_extractor(audio_gt)
        audio = self.model.decode(features)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio


class FreeVWrapper:
    def __init__(self, ckpt_path, config_path, device):
        if FREEV_REPO_PATH not in sys.path:
            sys.path.insert(0, FREEV_REPO_PATH)
        from models2 import Generator

        with open(config_path) as f:
            h_dict = json.load(f)

        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.__dict__ = self
        h = AttrDict(h_dict)
        self.h = h
        self.generator = Generator(h).to(device)
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)
        self.generator.eval()
        self.device = device
        self.sr = h.get('sampling_rate', 22050)
        n_params = sum(p.numel() for p in self.generator.parameters())
        print(f"  FreeV params: {n_params/1e6:.2f}M, sr={self.sr}")

    @torch.no_grad()
    def __call__(self, audio_gt):
        h = self.h
        mel = mel_spectrogram_hifigan(
            audio_gt,
            n_fft=h['n_fft'], num_mels=h['num_mels'], sampling_rate=h['sampling_rate'],
            hop_size=h['hop_size'], win_size=h['win_size'],
            fmin=h['fmin'], fmax=h['fmax'],
        )
        outputs = self.generator(mel)
        if isinstance(outputs, (list, tuple)):
            audio = outputs[-1]
        else:
            audio = outputs
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio


def load_audio(path, target_sr):
    audio_np, sr = sf.read(str(path), dtype='float32', always_2d=True)
    audio = torch.from_numpy(audio_np.T)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def crop_segment(audio, segment_size, mode="first", seed=None):
    """audio: (1, T). segment_size в сэмплах."""
    T = audio.shape[-1]
    if T <= segment_size:
        pad = segment_size - T
        return F.pad(audio, (0, pad))
    if mode == "first":
        start = 0
    elif mode == "center":
        start = (T - segment_size) // 2
    elif mode == "random":
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        start = int(torch.randint(0, T - segment_size + 1, (1,), generator=g).item())
    else:
        raise ValueError(mode)
    return audio[..., start:start + segment_size]


def get_test_files(metadata_path, wavs_dir, offset, limit, split_symbol="|"):
    wavs_dir = Path(wavs_dir)
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(split_symbol)
            file_id = parts[0]
            wav_path = wavs_dir / f"{file_id}.wav"
            entries.append(wav_path)
    sel = entries[offset:offset + limit]
    missing = [p for p in sel if not p.exists()]
    if missing:
        print(f"WARNING: {len(missing)} файлов не найдено, например: {missing[0]}")
        sel = [p for p in sel if p.exists()]
    return sel


def evaluate_model(model_name, wrapper, audio_files, device):
    results = defaultdict(list)
    sr = wrapper.sr

    seg_target = None
    if USE_SEGMENT:
        seg_target = int(round(SEGMENT_SIZE * sr / 22050))

    for idx, audio_path in enumerate(tqdm(audio_files, desc=f"  {model_name}")):
        try:
            audio_gt = load_audio(audio_path, sr).to(device)
            if seg_target is not None:
                audio_gt = crop_segment(audio_gt, seg_target,
                                        mode=SEGMENT_MODE, seed=idx)
            audio_pred = wrapper(audio_gt)
            if audio_pred.dim() == 1:
                audio_pred = audio_pred.unsqueeze(0)

            metrics = compute_all_metrics(audio_pred, audio_gt, sr)
            for k, v in metrics.items():
                if not np.isnan(v):
                    results[k].append(v)
        except Exception as e:
            print(f"\nError on {audio_path.name}: {e}")
            continue

    return {
        "mean": {k: float(np.mean(v)) for k, v in results.items()},
        "raw":  {k: [float(x) for x in v] for k, v in results.items()},
        "n":    {k: len(v) for k, v in results.items()},
    }


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    audio_files = get_test_files(METADATA_FILE, LJSPEECH_WAVS_DIR,
                                  offset=OFFSET, limit=LIMIT,
                                  split_symbol=SPLIT_SYMBOL)
    print(f"Тестовая выборка: {len(audio_files)} файлов "
          f"(offset={OFFSET}, limit={LIMIT})")
    if len(audio_files) > 0:
        print(f"  Первый: {audio_files[0].name}")
        print(f"  Последний: {audio_files[-1].name}")
    print(f"  Сегмент: {'ON' if USE_SEGMENT else 'OFF'} "
          f"(size={SEGMENT_SIZE} @22050, mode={SEGMENT_MODE})\n")

    all_results = {}

    runs = [
        ("hifigan_v1", lambda: HiFiGANWrapper(HIFIGAN_V1_CKPT, HIFIGAN_V1_CONFIG, device),
            "HiFi-GAN v1 (13M)"),
        ("hifigan_v2", lambda: HiFiGANWrapper(HIFIGAN_V2_CKPT, HIFIGAN_V2_CONFIG, device),
            "HiFi-GAN v2 (~0.9M)"),
        ("hifigan_v3", lambda: HiFiGANWrapper(HIFIGAN_V3_CKPT, HIFIGAN_V3_CONFIG, device),
            "HiFi-GAN v3 (~1.5M)"),
        ("vocos",      lambda: VocosWrapper(device),
            "Vocos"),
        ("freev",      lambda: FreeVWrapper(FREEV_CKPT, FREEV_CONFIG, device),
            "FreeV"),
    ]

    for name, builder, header in runs:
        if name not in MODELS_TO_EVAL:
            continue
        print(f"\n{header}")
        try:
            wrapper = builder()
            all_results[name] = evaluate_model(name, wrapper, audio_files, device)
            del wrapper
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"{'Model':<20} {'LSD ↓':<12} {'PESQ ↑':<12} {'STOI ↑':<12} {'MelDist ↓':<12}")
    for model_name, res in all_results.items():
        m = res["mean"]
        print(f"{model_name:<20} "
            f"{m.get('lsd', float('nan')):<12.4f} "
            f"{m.get('pesq', float('nan')):<12.4f} "
            f"{m.get('stoi', float('nan')):<12.4f} "
            f"{m.get('mel_distance', float('nan')):<12.4f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nРезультаты сохранены в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()