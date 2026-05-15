import warnings
import os
import sys
import time
import gc
import json
import torch
import numpy as np

warnings.filterwarnings("ignore")


BENCH_GPU = True
BENCH_CPU = True
BENCH_CPU_THREAD_SWEEP = True

PRINT_PER_DEVICE_SUMMARY = True
PRINT_CPU_THREADS_SUMMARY = True
PRINT_CROSS_MODEL_GPU = True
PRINT_CROSS_MODEL_CPU_BEST = True

SAVE_JSON = True
SAVE_RAW_TIMES = True

MODELS_TO_BENCH = [
    ("istft_wav",         "src.model.istftwav",         "ISTFTWav",        {}),
    ("istft_wav_snake",   "src.model.istftwav_snake",   "ISTFTWavSnake",   {}),
]

CHUNK_SIZES_MS = [20, 50, 100, 200, 500]
LOOKBACK_FRAMES = 16
STREAM_DURATION_SEC = 20.0
NUM_WARMUP_CHUNKS = 10

SAMPLE_RATE = 22050
HOP_LENGTH = 256
N_MELS = 80

CPU_THREAD_CONFIGS = [1, 2, 4, 8, 16, None]

CPU_DEFAULT_THREADS = None  

OUTPUT_JSON = "streaming_perf_my_models.json"

sys.path.insert(0, os.getcwd())


def build_generator(module_path, class_name, kwargs, device, compile_mode=None):
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    gen = cls(**kwargs).to(device)
    gen.eval()
    if hasattr(gen, "remove_weight_norm"):
        try:
            gen.remove_weight_norm()
        except Exception:
            pass

    if compile_mode and sys.platform != "win32":
        try:
            gen = torch.compile(gen, mode=compile_mode, fullgraph=False)
            print(f"    torch.compile({compile_mode}) enabled on {device.type}")
        except Exception as e:
            print(f"    torch.compile failed: {e}")
    return gen


class Timer:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda
        if is_cuda:
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.is_cuda:
            self.s.record()
        else:
            self.t0 = time.perf_counter()

    def stop(self):
        if self.is_cuda:
            self.e.record()
            torch.cuda.synchronize()
            return self.s.elapsed_time(self.e) / 1000.0
        return time.perf_counter() - self.t0


@torch.no_grad()
def streaming_benchmark(generator, sr, hop, n_mels,
                        chunk_size_ms, lookback_frames,
                        stream_duration_sec, num_warmup_chunks,
                        device, is_cuda):
    hop_ms = 1000.0 * hop / sr
    chunk_frames = max(1, int(round(chunk_size_ms / hop_ms)))
    chunk_duration_s = chunk_frames * hop / sr

    total_frames = int(stream_duration_sec * sr / hop) + lookback_frames + 10
    long_mel = torch.randn(1, n_mels, total_frames, device=device)
    T = long_mel.shape[-1]

    n_chunks = (T - lookback_frames) // chunk_frames
    if n_chunks < num_warmup_chunks + 5:
        return {"chunk_size_ms": chunk_size_ms, "error": "not enough frames"}

    for i in range(num_warmup_chunks):
        cs = lookback_frames + i * chunk_frames
        ce = cs + chunk_frames
        _ = generator(long_mel[..., cs - lookback_frames: ce])
    if is_cuda:
        torch.cuda.synchronize()

    timer = Timer(is_cuda)
    times = []
    wall_start = time.perf_counter()
    for ci in range(num_warmup_chunks, n_chunks - 1):
        cs = lookback_frames + ci * chunk_frames
        ce = cs + chunk_frames
        if ce > T:
            break
        mel_in = long_mel[..., cs - lookback_frames: ce]
        timer.start()
        _ = generator(mel_in)
        times.append(timer.stop())
    wall_end = time.perf_counter()

    t = np.array(times)
    audio_s = len(t) * chunk_duration_s

    result = {
        "chunk_size_ms_target": chunk_size_ms,
        "chunk_size_ms_actual": chunk_duration_s * 1000,
        "chunk_frames": chunk_frames,
        "lookback_frames": lookback_frames,
        "n_chunks_measured": len(t),

        "rtf_mean": float(t.mean() / chunk_duration_s),
        "rtf_std":  float(t.std()  / chunk_duration_s),
        "rtf_p50":  float(np.percentile(t, 50) / chunk_duration_s),
        "rtf_p95":  float(np.percentile(t, 95) / chunk_duration_s),
        "rtf_p99":  float(np.percentile(t, 99) / chunk_duration_s),
        "rtf_max":  float(t.max() / chunk_duration_s),

        "rtf_steady_state": float(t[1:].mean() / chunk_duration_s) if len(t) > 1 else None,

        "first_chunk_time_ms": float(t[0] * 1000),
        "chunk_time_mean_ms": float(t.mean() * 1000),
        "chunk_time_p95_ms":  float(np.percentile(t, 95) * 1000),
        "chunk_time_p99_ms":  float(np.percentile(t, 99) * 1000),

        "overall_rtf": float((wall_end - wall_start) / audio_s),

        "min_algorithmic_latency_ms": chunk_duration_s * 1000,
        "total_latency_p95_ms": float(chunk_duration_s * 1000 + np.percentile(t, 95) * 1000),

        "is_realtime_capable_mean": bool(t.mean() / chunk_duration_s < 1.0),
        "is_realtime_capable_p99":  bool(np.percentile(t, 99) / chunk_duration_s < 1.0),
    }
    if SAVE_RAW_TIMES:
        result["raw_times_s"] = t.tolist()
    return result


def run_for_device(name, module_path, class_name, kwargs, device_str, label=None):
    is_cuda = (device_str == "cuda")
    device = torch.device(device_str)

    tag = label if label else device_str.upper()
    print(f"\n  [{tag}]")
    try:
        gen = build_generator(
            module_path, class_name, kwargs, device,
            compile_mode=None,
        )
    except Exception as e:
        print(f"    Failed to build model on {device_str}: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e)}

    sr = getattr(gen, "sr", SAMPLE_RATE)
    hop = getattr(gen, "hop_length", HOP_LENGTH)
    n_mels = N_MELS

    num_params = sum(p.numel() for p in gen.parameters())
    print(f"    Params: {num_params / 1e6:.2f}M  sr={sr}  hop={hop}  n_mels={n_mels}")
    print(f"    Lookback: {LOOKBACK_FRAMES} frames = {LOOKBACK_FRAMES * hop * 1000 / sr:.1f}ms")
    if not is_cuda:
        print(f"    Torch num_threads: {torch.get_num_threads()}")
    print()

    if is_cuda:
        torch.backends.cudnn.benchmark = True

    results = {}
    for cs_ms in CHUNK_SIZES_MS:
        print(f"    chunk={cs_ms}ms ...", end=" ", flush=True)
        try:
            r = streaming_benchmark(
                gen, sr, hop, n_mels,
                cs_ms, LOOKBACK_FRAMES,
                STREAM_DURATION_SEC, NUM_WARMUP_CHUNKS,
                device, is_cuda,
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
            import traceback; traceback.print_exc()
            results[f"{cs_ms}ms"] = {"error": str(e)}

    del gen
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()
    return results


def print_summary(name, device_label, dev_res):
    print(f"\n{'='*120}")
    print(f"SUMMARY [{device_label.upper()}]  Model: {name}")
    print("="*120)
    print(f"{'Chunk':<10} {'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
          f"{'1st chunk ms':<14} {'Total lat ms':<14} {'RT capable':<14}")
    print("-"*120)
    if "error" in dev_res:
        print(f"  ERROR: {dev_res['error']}")
        return
    for cs_ms in CHUNK_SIZES_MS:
        r = dev_res.get(f"{cs_ms}ms", {})
        if not r or "error" in r:
            continue
        rt = "yes" if r.get("is_realtime_capable_p99") else "NO (p99>1)"
        print(f"{cs_ms}ms      "
              f"{r['rtf_mean']:<12.4f} {r['rtf_p95']:<12.4f} {r['rtf_p99']:<12.4f} "
              f"{r['first_chunk_time_ms']:<14.2f} "
              f"{r['total_latency_p95_ms']:<14.2f} "
              f"{rt:<14}")


def print_cpu_threads_summary(name, cpu_by_threads):
    print(f"\n{'='*120}")
    print(f"CPU THREADS SWEEP  Model: {name}")
    print("="*120)
    header = f"{'Threads':<10}" + "".join(f"{f'{c}ms p99':<14}" for c in CHUNK_SIZES_MS)
    print(header)
    print("-"*120)
    for thr_label, dev_res in cpu_by_threads.items():
        if "error" in dev_res:
            print(f"{thr_label:<10} ERROR")
            continue
        row = f"{thr_label:<10}"
        for cs_ms in CHUNK_SIZES_MS:
            r = dev_res.get(f"{cs_ms}ms", {})
            if not r or "error" in r:
                row += f"{'—':<14}"
            else:
                row += f"{r['rtf_p99']:<14.4f}"
        print(row)


def bench_gpu(name, module_path, class_name, kwargs):
    return run_for_device(name, module_path, class_name, kwargs, "cuda", label="CUDA")


def bench_cpu_sweep(name, module_path, class_name, kwargs, cpu_count):
    cpu_by_threads = {}
    for n_threads in CPU_THREAD_CONFIGS:
        if n_threads is None:
            actual = cpu_count
            thr_label = f"all({actual})"
        else:
            actual = n_threads
            thr_label = f"{n_threads}"

        torch.set_num_threads(actual)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        cpu_by_threads[thr_label] = run_for_device(
            name, module_path, class_name, kwargs, "cpu",
            label=f"CPU threads={thr_label}"
        )
    return cpu_by_threads


def bench_cpu_single(name, module_path, class_name, kwargs, cpu_count):
    if CPU_DEFAULT_THREADS is None:
        actual = cpu_count
        thr_label = f"all({actual})"
    else:
        actual = CPU_DEFAULT_THREADS
        thr_label = f"{actual}"

    torch.set_num_threads(actual)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    return run_for_device(
        name, module_path, class_name, kwargs, "cpu",
        label=f"CPU threads={thr_label}"
    )

def main():
    print("MKL:", torch.backends.mkl.is_available())
    print("MKLDNN:", torch.backends.mkldnn.is_available())
    print(torch.__config__.parallel_info())

    cpu_count = os.cpu_count()
    gpu_available = torch.cuda.is_available()
    do_gpu = BENCH_GPU and gpu_available
    do_cpu = BENCH_CPU

    print("="*80)
    print("STREAMING RTF BENCHMARK (random weights, architecture only)")
    print("="*80)
    print(f"Chunk sizes:        {CHUNK_SIZES_MS} ms")
    print(f"Lookback:           {LOOKBACK_FRAMES} frames")
    print(f"Stream duration:    {STREAM_DURATION_SEC}s")
    print(f"Warmup chunks:      {NUM_WARMUP_CHUNKS}")
    print(f"CPU count:          {cpu_count}")
    print(f"GPU available:      {gpu_available}")
    print(f"Models:             {[m[0] for m in MODELS_TO_BENCH]}")
    print()
    print("Flags:")
    print(f"  BENCH_GPU              = {BENCH_GPU}  (effective: {do_gpu})")
    print(f"  BENCH_CPU              = {BENCH_CPU}")
    print(f"  BENCH_CPU_THREAD_SWEEP = {BENCH_CPU_THREAD_SWEEP}")
    if do_cpu:
        if BENCH_CPU_THREAD_SWEEP:
            print(f"  CPU_THREAD_CONFIGS     = {CPU_THREAD_CONFIGS}")
        else:
            print(f"  CPU_DEFAULT_THREADS    = {CPU_DEFAULT_THREADS}")
    print(f"  SAVE_JSON              = {SAVE_JSON}")
    print(f"  SAVE_RAW_TIMES         = {SAVE_RAW_TIMES}")

    if not do_gpu and not do_cpu:
        print("\nNothing to bench — оба BENCH_GPU и BENCH_CPU выключены. Выход.")
        return

    all_results = {}

    for name, module_path, class_name, kwargs in MODELS_TO_BENCH:
        print(f"\n{'='*80}")
        print(f"MODEL: {name}  ({module_path}.{class_name})")
        print(f"{'='*80}")
        all_results[name] = {}

        if do_gpu:
            all_results[name]["gpu"] = bench_gpu(name, module_path, class_name, kwargs)
            if PRINT_PER_DEVICE_SUMMARY:
                print_summary(name, "gpu", all_results[name]["gpu"])

        if do_cpu:
            if BENCH_CPU_THREAD_SWEEP:
                cpu_by_threads = bench_cpu_sweep(name, module_path, class_name, kwargs, cpu_count)
                all_results[name]["cpu_by_threads"] = cpu_by_threads
                if PRINT_CPU_THREADS_SUMMARY:
                    print_cpu_threads_summary(name, cpu_by_threads)
            else:
                cpu_res = bench_cpu_single(name, module_path, class_name, kwargs, cpu_count)
                all_results[name]["cpu"] = cpu_res
                if PRINT_PER_DEVICE_SUMMARY:
                    print_summary(name, "cpu", cpu_res)

    if PRINT_CROSS_MODEL_GPU or PRINT_CROSS_MODEL_CPU_BEST:
        print(f"\n\n{'#'*120}")
        print("# CROSS-MODEL COMPARISON")
        print(f"{'#'*120}")

    if do_gpu and PRINT_CROSS_MODEL_GPU:
        print(f"\n{'='*120}")
        print(f"COMPARISON [GPU]")
        print(f"{'='*120}")
        print(f"{'Model':<22} {'Chunk':<10} {'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
              f"{'1st ms':<10} {'Total lat ms':<14} {'RT':<6}")
        print("-"*120)
        for name in [m[0] for m in MODELS_TO_BENCH]:
            dev_res = all_results.get(name, {}).get("gpu", {})
            if "error" in dev_res:
                print(f"{name:<22} ERROR: {dev_res['error'][:80]}")
                continue
            for cs_ms in CHUNK_SIZES_MS:
                r = dev_res.get(f"{cs_ms}ms", {})
                if not r or "error" in r:
                    continue
                rt = "yes" if r.get("is_realtime_capable_p99") else "NO"
                print(f"{name:<22} {cs_ms}ms      "
                      f"{r['rtf_mean']:<12.4f} {r['rtf_p95']:<12.4f} {r['rtf_p99']:<12.4f} "
                      f"{r['first_chunk_time_ms']:<10.2f} "
                      f"{r['total_latency_p95_ms']:<14.2f} "
                      f"{rt:<6}")
            print()

    if do_cpu and PRINT_CROSS_MODEL_CPU_BEST:
        print(f"\n{'='*120}")
        if BENCH_CPU_THREAD_SWEEP:
            print(f"COMPARISON [CPU best-of-threads]")
        else:
            print(f"COMPARISON [CPU]")
        print(f"{'='*120}")
        print(f"{'Model':<22} {'Chunk':<10} {'Best thr':<10} "
              f"{'RTF mean':<12} {'RTF p95':<12} {'RTF p99':<12} "
              f"{'1st ms':<10} {'Total lat ms':<14} {'RT':<6}")
        print("-"*120)
        for name in [m[0] for m in MODELS_TO_BENCH]:
            entry = all_results.get(name, {})
            if BENCH_CPU_THREAD_SWEEP:
                cpu_by_threads = entry.get("cpu_by_threads", {})
            else:
                single = entry.get("cpu", {})
                cpu_by_threads = {"single": single} if single else {}

            for cs_ms in CHUNK_SIZES_MS:
                best_thr = None
                best_r = None
                for thr_label, dev_res in cpu_by_threads.items():
                    if "error" in dev_res:
                        continue
                    r = dev_res.get(f"{cs_ms}ms", {})
                    if not r or "error" in r:
                        continue
                    if best_r is None or r["rtf_p99"] < best_r["rtf_p99"]:
                        best_r = r
                        best_thr = thr_label
                if best_r is None:
                    continue
                rt = "yes" if best_r.get("is_realtime_capable_p99") else "NO"
                print(f"{name:<22} {cs_ms}ms      {best_thr:<10} "
                      f"{best_r['rtf_mean']:<12.4f} {best_r['rtf_p95']:<12.4f} {best_r['rtf_p99']:<12.4f} "
                      f"{best_r['first_chunk_time_ms']:<10.2f} "
                      f"{best_r['total_latency_p95_ms']:<14.2f} "
                      f"{rt:<6}")
            print()

    print("="*120)
    print("Notes:")
    print("  RTF = chunk_inference_time / chunk_audio_duration  (< 1 = realtime)")
    print(f"  Total latency = chunk_size + p95 inference time")
    print(f"  Lookback {LOOKBACK_FRAMES} frames прошлого контекста")
    print("  Random weights — на скорость инференса значения весов не влияют")
    print("="*120)

    if SAVE_JSON:
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()