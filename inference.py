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


LJSPEECH_ROOT     = "E:/se/4_course/audio_vocoder/data/LJSpeech-1.1/LJSpeech-1.1"
LJSPEECH_WAVS_DIR = LJSPEECH_ROOT + "/wavs"
METADATA_FILE     = LJSPEECH_ROOT + "/metadata.csv"
SPLIT_SYMBOL      = "|"

PROJECT_ROOT = "E:/se/4_course/thesis/audio_vocoder"

OFFSET       = 12500
LIMIT        = 600
SEGMENT_SIZE = 16384
USE_SEGMENT  = True
SEGMENT_MODE = "first"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_JSON = "benchmark_results_my.json"


MODELS_TO_EVAL = [
    {
        "name":         "istft_wav",
        "module_path":  "src.model.istftwav",
        "class_name":   "ISTFTWav",
        "kwargs":       {},
        "ckpt_path":    "E:/se/4_course/thesis/audio_vocoder/weights/model_best_istftwav_2.pth",
    },
    {
        "name":         "istft_wav_snake",
        "module_path":  "src.model.istftwav_snake",
        "class_name":   "ISTFTWavSnake",
        "kwargs":       {},
        "ckpt_path":    "E:/se/4_course/thesis/audio_vocoder/weights/model_best_istftwav_snake_2.pth",
    },
    # {
    #     "name":         "without_istft_wav",
    #     "module_path":  "src.model.without_istftwav",
    #     "class_name":   "WithoutISTFTWav",
    #     "kwargs":       {},
    #     "ckpt_path":    "E:/se/4_course/thesis/audio_vocoder/saved_freev_baseline/ISTFTWav-fast-full-data/model_best.pth",
    # },
]


sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.getcwd())

from src.transforms.audio_transforms import AudioToMelSpectrogram   # noqa: E402


_metric_mel_basis = {}
_metric_hann_window = {}

def _dyn_range_compress(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram_hifigan(y, n_fft=1024, num_mels=80, sampling_rate=22050,
                             hop_size=256, win_size=1024, fmin=0, fmax=8000,
                             center=False):
    if y.dim() == 1:
        y = y.unsqueeze(0)

    key = f"{fmax}_{num_mels}_{sampling_rate}_{y.device}"
    if key not in _metric_mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels,
                             fmin=fmin, fmax=fmax)
        _metric_mel_basis[key] = torch.from_numpy(mel).float().to(y.device)
    win_key = f"{win_size}_{y.device}"
    if win_key not in _metric_hann_window:
        _metric_hann_window[win_key] = torch.hann_window(win_size).to(y.device)

    pad = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=_metric_hann_window[win_key],
                      center=center, pad_mode='reflect',
                      normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(_metric_mel_basis[key], spec)
    spec = _dyn_range_compress(spec)
    return spec


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


class MyModelWrapper:
    def __init__(self, module_path, class_name, kwargs, ckpt_path, device):
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self.generator = cls(**kwargs).to(device)

        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            full_sd = state["state_dict"]
        else:
            full_sd = state

        GEN_PREFIX = "generator."
        gen_sd = {
            k[len(GEN_PREFIX):]: v
            for k, v in full_sd.items()
            if k.startswith(GEN_PREFIX)
        }
        if len(gen_sd) == 0:
            available = set(k.split('.')[0] for k in full_sd.keys())
            raise RuntimeError(
                f"Не нашёл ключей '{GEN_PREFIX}' в {ckpt_path}. Префиксы: {available}"
            )

        missing, unexpected = self.generator.load_state_dict(gen_sd, strict=True)
        print(f"  Loaded {len(gen_sd)} generator weights from epoch="
              f"{state.get('epoch', '?')}, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

        self.generator.eval()
        if hasattr(self.generator, "remove_weight_norm"):
            try:
                self.generator.remove_weight_norm()
            except Exception:
                pass

        self.device = device
        self.sr = getattr(self.generator, "sr", 22050)
        self.hop_length = getattr(self.generator, "hop_length", 256)
        self.n_mels = 80
        self.n_fft = 1024
        self.win_length = 1024
        self.f_min = 0.0
        self.f_max = 8000.0

        self.mel_transform = AudioToMelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
        )
        self.mel_transform.mel_transform = self.mel_transform.mel_transform.to(device)

        n_params = sum(p.numel() for p in self.generator.parameters())
        print(f"  Loaded {class_name} from {ckpt_path}")
        print(f"  Params: {n_params/1e6:.2f}M, sr={self.sr}, hop={self.hop_length}")

        self._dbg_printed = False

    @torch.no_grad()
    def __call__(self, audio_gt):
        if audio_gt.dim() == 1:
            audio_in = audio_gt.unsqueeze(0)
        else:
            audio_in = audio_gt

        # AudioToMelSpectrogram возвращает [C, n_mels, T_mel] (т.к. внутри unsqueeze(0))
        mel = self.mel_transform(audio_in)
        # Приводим к [B, n_mels, T_mel]
        if mel.dim() == 4:           # [B, C, n_mels, T_mel]
            mel = mel.squeeze(1)
        elif mel.dim() == 2:         # [n_mels, T_mel] — на всякий
            mel = mel.unsqueeze(0)
        # сейчас mel: [B, n_mels, T_mel]

        audio = self.generator(mel)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        if not self._dbg_printed:
            print(f"  [debug] mel  shape={tuple(mel.shape)}  "
                  f"min={mel.min().item():.3f}  max={mel.max().item():.3f}  "
                  f"mean={mel.mean().item():.3f}")
            print(f"  [debug] pred shape={tuple(audio.shape)}  "
                  f"min={audio.min().item():.3f}  max={audio.max().item():.3f}")
            print(f"  [debug] gt   shape={tuple(audio_in.shape)}  "
                  f"min={audio_in.min().item():.3f}  max={audio_in.max().item():.3f}")
            self._dbg_printed = True

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
        print(f"WARNING: {len(missing)} файлов не найдено")
        sel = [p for p in sel if p.exists()]
    return sel


def evaluate_model(model_name, wrapper, audio_files, device):
    results = defaultdict(list)
    nan_counts = defaultdict(int)
    sr = wrapper.sr

    seg_target = None
    if USE_SEGMENT:
        seg_target = int(round(SEGMENT_SIZE * sr / 22050))

    for idx, audio_path in enumerate(tqdm(audio_files, desc=f"  {model_name}")):
        try:
            audio_gt = load_audio(audio_path, sr).to(device)         # [1, T]
            if seg_target is not None:
                audio_gt = crop_segment(audio_gt, seg_target,
                                        mode=SEGMENT_MODE, seed=idx)
            audio_pred = wrapper(audio_gt)
            if audio_pred.dim() == 1:
                audio_pred = audio_pred.unsqueeze(0)

            metrics = compute_all_metrics(audio_pred, audio_gt, sr)
            for k, v in metrics.items():
                if np.isnan(v):
                    nan_counts[k] += 1
                else:
                    results[k].append(v)
        except Exception as e:
            print(f"\nError on {audio_path.name}: {e}")
            import traceback; traceback.print_exc()
            continue

    print(f"  nan counts: {dict(nan_counts)}")
    return {
        "mean": {k: float(np.mean(v)) for k, v in results.items()},
        "raw":  {k: [float(x) for x in v] for k, v in results.items()},
        "n":    {k: len(v) for k, v in results.items()},
        "nan_count": dict(nan_counts),
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

    for entry in MODELS_TO_EVAL:
        name = entry["name"]
        print(f"\n=== {name} ===")
        try:
            wrapper = MyModelWrapper(
                module_path = entry["module_path"],
                class_name  = entry["class_name"],
                kwargs      = entry["kwargs"],
                ckpt_path   = entry["ckpt_path"],
                device      = device,
            )
            all_results[name] = evaluate_model(name, wrapper, audio_files, device)
            del wrapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results[name] = {"error": str(e)}

    print("\n" + "=" * 80)
    print(f"{'Model':<22} {'LSD ↓':<12} {'PESQ ↑':<12} {'STOI ↑':<12} {'MelDist ↓':<12}")
    print("=" * 80)
    for model_name, res in all_results.items():
        if "error" in res:
            print(f"{model_name:<22} ERROR")
            continue
        m = res["mean"]
        print(f"{model_name:<22} "
              f"{m.get('lsd', float('nan')):<12.4f} "
              f"{m.get('pesq', float('nan')):<12.4f} "
              f"{m.get('stoi', float('nan')):<12.4f} "
              f"{m.get('mel_distance', float('nan')):<12.4f}")
    print("=" * 80)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nРезультаты сохранены в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()