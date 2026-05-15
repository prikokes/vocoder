import os
import sys
import json
import time
import gc
import torch
import numpy as np
import torch.nn.functional as F
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

CHUNK_SIZES_MS = [20, 50, 100, 200, 500]

LOOKBACK_FRAMES = 16
STREAM_DURATION_SEC = 20.0
NUM_WARMUP_CHUNKS = 10

BENCH_GPU = torch.cuda.is_available()
BENCH_CPU = True

CPU_THREADS_TO_TRY = [1, 2, 4, 8, 16, os.cpu_count()]

OUTPUT_JSON = "streaming_perf_results.json"


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
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.generator.load_state_dict(sd['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        self.device = device
        self.sr = h.sampling_rate
        self.hop_size = h.hop_size
        self.num_mels = h.num_mels

    def make_long_mel(self, duration_sec):
        n_samples = int(duration_sec * self.sr)
        audio = torch.randn(1, n_samples, device=self.device)
        h = self.h
        return mel_spectrogram_hifigan(
            audio, n_fft=h.n_fft, num_mels=h.num_mels,
            sampling_rate=h.sampling_rate, hop_size=h.hop_size,
            win_size=h.win_size, fmin=h.fmin, fmax=h.fmax,
        )

    @torch.no_grad()
    def infer(self, mel_chunk):
        return self.generator(mel_chunk)


class VocosWrapper:
    def __init__(self, device):
        from vocos import Vocos
        self.model = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
        self.model.eval()
        self.device = device
        self.sr = 24000
        self.hop_size = 256
        self.num_mels = 100

    def make_long_mel(self, duration_sec):
        n_samples = int(duration_sec * self.sr)
        audio = torch.randn(1, n_samples, device=self.device)
        with torch.no_grad():
            features = self.model.feature_extractor(audio)
        return features

    @torch.no_grad()
    def infer(self, features_chunk):
        return self.model.decode(features_chunk)


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
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'generator' in sd:
            self.generator.load_state_dict(sd['generator'])
        else:
            self.generator.load_state_dict(sd)
        self.generator.eval()
        self.device = device
        self.sr = h.get('sampling_rate', 22050)
        self.hop_size = h.get('hop_size', 256)
        self.num_mels = h.get('num_mels', 80)

    def make_long_mel(self, duration_sec):
        n_samples = int(duration_sec * self.sr)
        audio = torch.randn(1, n_samples, device=self.device)
        h = self.h
        return mel_spectrogram_hifigan(
            audio, n_fft=h['n_fft'], num_mels=h['num_mels'],
            sampling_rate=h['sampling_rate'], hop_size=h['hop_size'],
            win_size=h['win_size'], fmin=h['fmin'], fmax=h['fmax'],
        )

    @torch.no_grad()
    def infer(self, mel_chunk):
        out = self.generator(mel_chunk)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out


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


class Timer:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda
        if is_cuda:
            self.start_ev = torch.cuda.Event(enable_timing=True)
            self.end_ev   = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.is_cuda:
            self.start_ev.record()
        else:
            self.t0 = time.perf_counter()

    def stop(self):
        if self.is_cuda:
            self.end_ev.record()
            torch.cuda.synchronize()
            return self.start_ev.elapsed_time(self.end_ev) / 1000.0
        else:
            return time.perf_counter() - self.t0


def streaming_benchmark(wrapper, chunk_size_ms, lookback_frames,
                         stream_duration_sec, num_warmup_chunks, is_cuda):
    sr = wrapper.sr
    hop = wrapper.hop_size

    hop_ms = 1000.0 * hop / sr
    chunk_frames = max(1, int(round(chunk_size_ms / hop_ms)))
    chunk_duration_s = chunk_frames * hop / sr

    long_mel = wrapper.make_long_mel(stream_duration_sec)
    T_total = long_mel.shape[-1]

    available_frames = T_total - lookback_frames
    n_chunks = available_frames // chunk_frames
    if n_chunks < num_warmup_chunks + 5:
        return {
            "chunk_size_ms": chunk_size_ms,
            "chunk_frames": chunk_frames,
            "error": f"not enough frames: T_total={T_total}, n_chunks={n_chunks}"
        }

    for i in range(num_warmup_chunks):
        s = lookback_frames + i * chunk_frames - lookback_frames
        e = lookback_frames + i * chunk_frames + chunk_frames
        s = max(0, s)
        mel_in = long_mel[..., s:e]
        _ = wrapper.infer(mel_in)
    if is_cuda:
        torch.cuda.synchronize()

    timer = Timer(is_cuda)
    per_chunk_times = []

    wall_start = time.perf_counter()

    for ci in range(num_warmup_chunks, num_warmup_chunks + n_chunks - num_warmup_chunks - 1):
        chunk_start = lookback_frames + ci * chunk_frames
        chunk_end   = chunk_start + chunk_frames
        if chunk_end > T_total:
            break
        mel_in = long_mel[..., chunk_start - lookback_frames : chunk_end]

        timer.start()
        audio_out = wrapper.infer(mel_in)
        if audio_out.dim() == 3:
            audio_out = audio_out.squeeze(1)
        dt = timer.stop()
        per_chunk_times.append(dt)

    wall_end = time.perf_counter()

    times = np.array(per_chunk_times)
    audio_processed_s = len(times) * chunk_duration_s

    return {
        "chunk_size_ms_target": chunk_size_ms,
        "chunk_size_ms_actual": chunk_duration_s * 1000,
        "chunk_frames": chunk_frames,
        "lookback_frames": lookback_frames,
        "n_chunks_measured": len(times),

        "rtf_mean": float(times.mean() / chunk_duration_s),
        "rtf_std":  float(times.std()  / chunk_duration_s),
        "rtf_p50":  float(np.percentile(times, 50) / chunk_duration_s),
        "rtf_p95":  float(np.percentile(times, 95) / chunk_duration_s),
        "rtf_p99":  float(np.percentile(times, 99) / chunk_duration_s),
        "rtf_max":  float(times.max() / chunk_duration_s),

        "rtf_steady_state": float(times[1:].mean() / chunk_duration_s) if len(times) > 1 else None,

        "first_chunk_time_ms": float(times[0] * 1000),
        "first_chunk_rtf": float(times[0] / chunk_duration_s),

        "chunk_time_mean_ms": float(times.mean() * 1000),
        "chunk_time_std_ms":  float(times.std()  * 1000),
        "chunk_time_p95_ms":  float(np.percentile(times, 95) * 1000),
        "chunk_time_p99_ms":  float(np.percentile(times, 99) * 1000),

        "overall_rtf": float((wall_end - wall_start) / audio_processed_s),

        "min_algorithmic_latency_ms": chunk_duration_s * 1000,
        "total_latency_p95_ms": float(chunk_duration_s * 1000 + np.percentile(times, 95) * 1000),

        "is_realtime_capable_mean": bool(times.mean() / chunk_duration_s < 1.0),
        "is_realtime_capable_p99":  bool(np.percentile(times, 99) / chunk_duration_s < 1.0),
        "raw_times_s": times.tolist(),
    }


def run_chunks_for_wrapper(wrapper, is_cuda):
    results = {}
    for cs_ms in CHUNK_SIZES_MS:
        print(f"      chunk={cs_ms}ms ...", end=" ", flush=True)
        try:
            r = streaming_benchmark(
                wrapper, cs_ms, LOOKBACK_FRAMES,
                STREAM_DURATION_SEC, NUM_WARMUP_CHUNKS, is_cuda
            )
            results[f"{cs_ms}ms"] = r
            if "error" in r:
                print(f"SKIP ({r['error']})")
            else:
                rt = "✓" if r["is_realtime_capable_p99"] else "✗p99"
                print(f"RTF mean={r['rtf_mean']:.4f} p95={r['rtf_p95']:.4f} "
                      f"p99={r['rtf_p99']:.4f}  first={r['first_chunk_time_ms']:.1f}ms  {rt}")
        except Exception as e:
            print(f"FAIL: {e}")
            results[f"{cs_ms}ms"] = {"error": str(e)}
    return results


def run_gpu(name):
    device = torch.device("cuda")
    print(f"\n  [GPU]")
    try:
        wrapper = build_wrapper(name, device)
        print(f"    sr={wrapper.sr}, hop={wrapper.hop_size}, num_mels={wrapper.num_mels}")
        print(f"    lookback={LOOKBACK_FRAMES} frames = "
              f"{LOOKBACK_FRAMES * wrapper.hop_size * 1000 / wrapper.sr:.1f} ms\n")
        results = run_chunks_for_wrapper(wrapper, is_cuda=True)
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        print(f"    GPU FAILED: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}


def run_cpu_with_threads(name):
    device = torch.device("cpu")
    print(f"\n  [CPU] (перебор threads: {CPU_THREADS_TO_TRY})")
    
    per_threads = {}
    
    for n_threads in CPU_THREADS_TO_TRY:
        torch.set_num_threads(n_threads)
        print(f"\n threads={n_threads} (actual torch.get_num_threads()={torch.get_num_threads()})")
        try:
            wrapper = build_wrapper(name, device)
            results = run_chunks_for_wrapper(wrapper, is_cuda=False)
            per_threads[n_threads] = results
            del wrapper
            gc.collect()
        except Exception as e:
            print(f"    threads={n_threads} FAILED: {e}")
            import traceback; traceback.print_exc()
            per_threads[n_threads] = {"error": str(e)}
    
    best_per_chunk = {}
    for cs_ms in CHUNK_SIZES_MS:
        key = f"{cs_ms}ms"
        best_t, best_p99 = None, float("inf")
        for n_threads in CPU_THREADS_TO_TRY:
            r = per_threads.get(n_threads, {})
            if "error" in r:
                continue
            rc = r.get(key, {})
            if "error" in rc or not rc:
                continue
            p99 = rc.get("rtf_p99", float("inf"))
            if p99 < best_p99:
                best_p99 = p99
                best_t = n_threads
        if best_t is not None:
            best_per_chunk[key] = {
                "best_threads": best_t,
                "rtf_p99": best_p99,
                "result": per_threads[best_t][key],
            }
    
    return {
        "per_threads": per_threads,
        "best_per_chunk": best_per_chunk,
    }


def print_cpu_summary_for_threads(all_results, n_threads):
    print(f"SUMMARY: CPU STREAMING (threads = {n_threads})")
    print(f"{'Model':<14} {'Chunk':<10} {'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
          f"{'1st chunk ms':<14} {'Total lat ms':<14} {'RT capable':<12}")
    for name in MODELS_TO_EVAL:
        cpu_res = all_results.get(name, {}).get("cpu", {})
        if "error" in cpu_res:
            print(f"{name:<14} ERROR: {cpu_res['error']}")
            continue
        per_t = cpu_res.get("per_threads", {}).get(n_threads, {})
        if not per_t or "error" in per_t:
            print(f"{name:<14} (no data for threads={n_threads})")
            continue
        for cs_ms in CHUNK_SIZES_MS:
            r = per_t.get(f"{cs_ms}ms", {})
            if "error" in r or not r:
                continue
            rt = "yes" if r.get("is_realtime_capable_p99") else "NO (p99>1)"
            print(f"{name:<14} {cs_ms}ms      "
                  f"{r['rtf_mean']:<12.4f} {r['rtf_p95']:<12.4f} {r['rtf_p99']:<12.4f} "
                  f"{r['first_chunk_time_ms']:<14.2f} "
                  f"{r['total_latency_p95_ms']:<14.2f} "
                  f"{rt:<12}")
        print()


def print_cpu_best_summary(all_results):
    print(f"SUMMARY: CPU STREAMING (BEST-OF-THREADS per (model, chunk))")
    print(f"{'Model':<14} {'Chunk':<10} {'Best thr':<10} {'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
          f"{'1st chunk ms':<14} {'Total lat ms':<14} {'RT capable':<12}")
    for name in MODELS_TO_EVAL:
        cpu_res = all_results.get(name, {}).get("cpu", {})
        if "error" in cpu_res:
            print(f"{name:<14} ERROR: {cpu_res['error']}")
            continue
        best_pc = cpu_res.get("best_per_chunk", {})
        for cs_ms in CHUNK_SIZES_MS:
            entry = best_pc.get(f"{cs_ms}ms")
            if entry is None:
                continue
            r = entry["result"]
            rt = "yes" if r.get("is_realtime_capable_p99") else "NO (p99>1)"
            print(f"{name:<14} {cs_ms}ms      {entry['best_threads']:<10} "
                  f"{r['rtf_mean']:<12.4f} {r['rtf_p95']:<12.4f} {r['rtf_p99']:<12.4f} "
                  f"{r['first_chunk_time_ms']:<14.2f} "
                  f"{r['total_latency_p95_ms']:<14.2f} "
                  f"{rt:<12}")
        print()


def main():
    print("STREAMING RTF BENCHMARK (with CPU thread sweep)")
    print(f"Chunk sizes: {CHUNK_SIZES_MS} ms")
    print(f"CPU threads to try: {CPU_THREADS_TO_TRY}")
    print(f"Lookback context: {LOOKBACK_FRAMES} frames")
    print(f"Stream duration: {STREAM_DURATION_SEC}s")
    print(f"Warmup chunks: {NUM_WARMUP_CHUNKS}")
    print(f"GPU available: {BENCH_GPU}")
    print(f"CPU bench: {BENCH_CPU}")
    print(f"Models: {MODELS_TO_EVAL}")

    all_results = {}

    for name in MODELS_TO_EVAL:
        print(f"MODEL: {name}")
        all_results[name] = {}

        if BENCH_GPU:
            all_results[name]["gpu"] = run_gpu(name)
        if BENCH_CPU:
            all_results[name]["cpu"] = run_cpu_with_threads(name)

    print("SUMMARY: GPU STREAMING")
    print(f"{'Model':<14} {'Chunk':<10} {'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
          f"{'1st chunk ms':<14} {'Total lat ms':<14} {'RT capable':<12}")
    for name in MODELS_TO_EVAL:
        gpu_res = all_results.get(name, {}).get("gpu", {})
        if "error" in gpu_res:
            print(f"{name:<14} ERROR: {gpu_res['error']}")
            continue
        for cs_ms in CHUNK_SIZES_MS:
            r = gpu_res.get(f"{cs_ms}ms", {})
            if "error" in r or not r:
                continue
            rt = "yes" if r.get("is_realtime_capable_p99") else "NO (p99>1)"
            print(f"{name:<14} {cs_ms}ms      "
                  f"{r['rtf_mean']:<12.4f} {r['rtf_p95']:<12.4f} {r['rtf_p99']:<12.4f} "
                  f"{r['first_chunk_time_ms']:<14.2f} "
                  f"{r['total_latency_p95_ms']:<14.2f} "
                  f"{rt:<12}")
        print()

    if BENCH_CPU:
        for n_threads in CPU_THREADS_TO_TRY:
            print_cpu_summary_for_threads(all_results, n_threads)

        print_cpu_best_summary(all_results)

    print("Notes:")
    print("  RTF = chunk_inference_time / chunk_audio_duration")
    print("  RTF < 1 => укладывается в realtime")
    print("  p95/p99 показывают worst-case (важно для стриминга, без дропов)")
    print(f"  Total latency = chunk_size + chunk_inference_time(p95)")
    print(f"  Lookback: {LOOKBACK_FRAMES} frames прошлого контекста")
    print(f"  CPU threads tried: {CPU_THREADS_TO_TRY}")
    print(f"  'BEST-OF-THREADS' таблица показывает лучший thread по rtf_p99 для каждой пары (model, chunk)")
    print(f"  Отдельные per-thread таблицы — для production multi-stream сценария (1 thread/stream)")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nРезультаты сохранены в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()