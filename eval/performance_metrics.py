import os
import sys
import json
import time
import gc
import torch
import torchaudio
import numpy as np
import torch.nn.functional as F
from pathlib import Path
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

AUDIO_DURATION_SEC = 5.0

NUM_WARMUP = 5
NUM_RUNS   = 30

BENCH_GPU = torch.cuda.is_available()
BENCH_CPU = True

OUTPUT_JSON = "perf_results.json"

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

    def make_input(self, duration_sec):
        h = self.h
        n_samples = int(duration_sec * h.sampling_rate)
        audio = torch.randn(1, n_samples, device=self.device)
        mel = mel_spectrogram_hifigan(
            audio,
            n_fft=h.n_fft, num_mels=h.num_mels, sampling_rate=h.sampling_rate,
            hop_size=h.hop_size, win_size=h.win_size,
            fmin=h.fmin, fmax=h.fmax,
        )
        return mel

    @torch.no_grad()
    def infer(self, mel_input):
        return self.generator(mel_input)

    def get_model(self):
        return self.generator


class VocosWrapper:
    def __init__(self, device):
        from vocos import Vocos
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        self.model.eval()
        self.device = device
        self.sr = 24000

    def make_input(self, duration_sec):
        n_samples = int(duration_sec * self.sr)
        audio = torch.randn(1, n_samples, device=self.device)
        with torch.no_grad():
            features = self.model.feature_extractor(audio)
        return features

    @torch.no_grad()
    def infer(self, features):
        return self.model.decode(features)

    def get_model(self):
        return self.model


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

    def make_input(self, duration_sec):
        h = self.h
        n_samples = int(duration_sec * h['sampling_rate'])
        audio = torch.randn(1, n_samples, device=self.device)
        mel = mel_spectrogram_hifigan(
            audio,
            n_fft=h['n_fft'], num_mels=h['num_mels'], sampling_rate=h['sampling_rate'],
            hop_size=h['hop_size'], win_size=h['win_size'],
            fmin=h['fmin'], fmax=h['fmax'],
        )
        return mel

    @torch.no_grad()
    def infer(self, mel_input):
        return self.generator(mel_input)

    def get_model(self):
        return self.generator



def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_macs(model, dummy_input):
    try:
        from thop import profile
        model.eval()
        macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return int(macs), "thop"
    except ImportError:
        pass
    except Exception as e:
        thop_err = str(e)
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, dummy_input).total()
        return int(flops // 2), "fvcore"
    except ImportError:
        return None, "thop/fvcore не установлены"
    except Exception as e:
        return None, f"fvcore error: {e}"


def benchmark_rtf(wrapper, device, duration_sec, num_warmup, num_runs, is_cuda):
    dummy_input = wrapper.make_input(duration_sec)

    # Warmup
    for _ in range(num_warmup):
        _ = wrapper.infer(dummy_input)
    if is_cuda:
        torch.cuda.synchronize()

    times = []
    if is_cuda:
        for _ in range(num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = wrapper.infer(dummy_input)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1000.0)
    else:
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = wrapper.infer(dummy_input)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    mean_t = times.mean()
    std_t  = times.std()
    rtf_mean = mean_t / duration_sec
    rtf_std  = std_t  / duration_sec
    return {
        "infer_time_mean_s": float(mean_t),
        "infer_time_std_s":  float(std_t),
        "rtf_mean": float(rtf_mean),
        "rtf_std":  float(rtf_std),
        "audio_duration_s": duration_sec,
        "num_runs": num_runs,
    }


def build_wrapper(name, device):
    if name == "hifigan_v1":
        return HiFiGANWrapper(HIFIGAN_V1_CKPT, HIFIGAN_V1_CONFIG, device)
    if name == "hifigan_v2":
        return HiFiGANWrapper(HIFIGAN_V2_CKPT, HIFIGAN_V2_CONFIG, device)
    if name == "hifigan_v3":
        return HiFiGANWrapper(HIFIGAN_V3_CKPT, HIFIGAN_V3_CONFIG, device)
    if name == "vocos":
        return VocosWrapper(device)
    if name == "freev":
        return FreeVWrapper(FREEV_CKPT, FREEV_CONFIG, device)
    raise ValueError(name)


def main():
    print(f"Audio duration for RTF: {AUDIO_DURATION_SEC}s")
    print(f"Warmup: {NUM_WARMUP}, Runs: {NUM_RUNS}")
    print(f"GPU available: {BENCH_GPU}")
    print(f"CPU bench: {BENCH_CPU}")
    print(f"PyTorch threads (CPU): {torch.get_num_threads()}\n")

    results = {}

    for name in MODELS_TO_EVAL:
        print(f"{name}")
        results[name] = {}

        if BENCH_GPU:
            print("\n[GPU]")
            try:
                device = torch.device("cuda")
                wrapper = build_wrapper(name, device)
                model = wrapper.get_model()

                total, trainable = count_params(model)
                print(f"  Params: total={total/1e6:.3f}M, trainable={trainable/1e6:.3f}M")

                dummy = wrapper.make_input(AUDIO_DURATION_SEC)
                macs, src = count_macs(model, dummy)
                if macs is not None:
                    gmacs = macs / 1e9
                    gmacs_per_sec = gmacs / AUDIO_DURATION_SEC
                    print(f"  MACs: {gmacs:.3f} G (для {AUDIO_DURATION_SEC}s аудио)  [{src}]")
                    print(f"  MACs/sec: {gmacs_per_sec:.3f} G/s аудио")
                else:
                    gmacs = None
                    gmacs_per_sec = None
                    print(f"  MACs: не посчитано ({src})")

                bench = benchmark_rtf(wrapper, device, AUDIO_DURATION_SEC,
                                      NUM_WARMUP, NUM_RUNS, is_cuda=True)
                rtf = bench["rtf_mean"]
                speedup = 1.0 / rtf if rtf > 0 else float('inf')
                print(f"  Time/inference: {bench['infer_time_mean_s']*1000:.2f} ± "
                      f"{bench['infer_time_std_s']*1000:.2f} ms")
                print(f"  RTF: {rtf:.5f} (faster than real-time x{speedup:.1f})")

                results[name]["gpu"] = {
                    "params_total": total,
                    "params_trainable": trainable,
                    "macs": macs,
                    "macs_per_sec_audio": (macs / AUDIO_DURATION_SEC) if macs else None,
                    "macs_source": src,
                    **bench,
                    "speedup_vs_realtime": speedup,
                }

                del wrapper, model, dummy
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  GPU FAILED: {e}")
                import traceback; traceback.print_exc()
                results[name]["gpu"] = {"error": str(e)}

        if BENCH_CPU:
            print("\n[CPU]")
            try:
                device = torch.device("cpu")
                wrapper = build_wrapper(name, device)
                model = wrapper.get_model()

                total, trainable = count_params(model)
                print(f"  Params: total={total/1e6:.3f}M, trainable={trainable/1e6:.3f}M")

                cpu_runs = max(5, NUM_RUNS // 3)
                bench = benchmark_rtf(wrapper, device, AUDIO_DURATION_SEC,
                                      NUM_WARMUP, cpu_runs, is_cuda=False)
                rtf = bench["rtf_mean"]
                speedup = 1.0 / rtf if rtf > 0 else float('inf')
                print(f"  Time/inference: {bench['infer_time_mean_s']*1000:.2f} ± "
                      f"{bench['infer_time_std_s']*1000:.2f} ms  ({cpu_runs} runs)")
                print(f"  RTF: {rtf:.5f} ({'faster' if rtf < 1 else 'slower'} "
                      f"than real-time x{speedup:.2f})")

                results[name]["cpu"] = {
                    "params_total": total,
                    "params_trainable": trainable,
                    **bench,
                    "speedup_vs_realtime": speedup,
                }

                del wrapper, model
                gc.collect()
            except Exception as e:
                print(f"  CPU FAILED: {e}")
                import traceback; traceback.print_exc()
                results[name]["cpu"] = {"error": str(e)}

    print("SUMMARY")
    header = f"{'Model':<14} {'Params (M)':<12} {'MACs (G)':<12} {'GPU RTF':<12} {'GPU x rt':<10} {'CPU RTF':<12} {'CPU x rt':<10}"
    print(header)
    for name in MODELS_TO_EVAL:
        r = results.get(name, {})
        gpu = r.get("gpu", {})
        cpu = r.get("cpu", {})
        params = gpu.get("params_total") or cpu.get("params_total")
        params_str = f"{params/1e6:.2f}" if params else "N/A"
        macs = gpu.get("macs")
        macs_str = f"{macs/1e9:.2f}" if macs else "N/A"
        gpu_rtf = gpu.get("rtf_mean")
        gpu_rtf_str = f"{gpu_rtf:.4f}" if gpu_rtf else "N/A"
        gpu_x = gpu.get("speedup_vs_realtime")
        gpu_x_str = f"x{gpu_x:.1f}" if gpu_x else "N/A"
        cpu_rtf = cpu.get("rtf_mean")
        cpu_rtf_str = f"{cpu_rtf:.4f}" if cpu_rtf else "N/A"
        cpu_x = cpu.get("speedup_vs_realtime")
        cpu_x_str = f"x{cpu_x:.2f}" if cpu_x else "N/A"
        print(f"{name:<14} {params_str:<12} {macs_str:<12} {gpu_rtf_str:<12} {gpu_x_str:<10} {cpu_rtf_str:<12} {cpu_x_str:<10}")
    print("RTF = inference_time / audio_duration. RTF < 1 => быстрее реалтайма.")
    print(f"Audio duration used: {AUDIO_DURATION_SEC}s")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nРезультаты сохранены в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()