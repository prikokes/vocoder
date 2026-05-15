import json
import numpy as np
from scipy import stats
from pathlib import Path

BASELINES_JSON = "streaming_perf_results.json"
MY_JSON        = "streaming_perf_my_models.json"
OUTPUT_JSON    = "streaming_stats.json"

CONFIDENCE   = 0.95
BOOTSTRAP_N  = 10_000
RNG          = np.random.default_rng(42)

MY_MODEL_NAMES = ["istft_wav", "istft_wav_snake"]
BASELINE_NAMES = ["hifigan_v1", "hifigan_v2", "hifigan_v3", "vocos", "freev"]

CHUNKS_MS = [20, 50, 100, 200, 500]


def _iter_runs_baseline(data):
    for model_name, by_dev in data.items():
        gpu = by_dev.get("gpu", {})
        if isinstance(gpu, dict) and "error" not in gpu:
            for chunk_key, run in gpu.items():
                if isinstance(run, dict) and "raw_times_s" in run:
                    yield {
                        "model": model_name,
                        "device": "gpu",
                        "threads": None,
                        "chunk_ms": int(chunk_key.replace("ms", "")),
                        "run": run,
                    }
        cpu = by_dev.get("cpu", {})
        if isinstance(cpu, dict) and "error" not in cpu:
            per_t = cpu.get("per_threads", {})
            for thr, by_chunk in per_t.items():
                if not isinstance(by_chunk, dict) or "error" in by_chunk:
                    continue
                for chunk_key, run in by_chunk.items():
                    if isinstance(run, dict) and "raw_times_s" in run:
                        yield {
                            "model": model_name,
                            "device": "cpu",
                            "threads": str(thr),
                            "chunk_ms": int(chunk_key.replace("ms", "")),
                            "run": run,
                        }


def _iter_runs_my(data):
    for model_name, by_dev in data.items():
        gpu = by_dev.get("gpu", {})
        if isinstance(gpu, dict) and "error" not in gpu:
            for chunk_key, run in gpu.items():
                if isinstance(run, dict) and "raw_times_s" in run:
                    yield {
                        "model": model_name,
                        "device": "gpu",
                        "threads": None,
                        "chunk_ms": int(chunk_key.replace("ms", "")),
                        "run": run,
                    }
        cpu_bt = by_dev.get("cpu_by_threads", {})
        for thr, by_chunk in cpu_bt.items():
            if not isinstance(by_chunk, dict) or "error" in by_chunk:
                continue
            for chunk_key, run in by_chunk.items():
                if isinstance(run, dict) and "raw_times_s" in run:
                    yield {
                        "model": model_name,
                        "device": "cpu",
                        "threads": str(thr),
                        "chunk_ms": int(chunk_key.replace("ms", "")),
                        "run": run,
                    }


def load_all_runs():
    records = []
    if Path(BASELINES_JSON).exists():
        with open(BASELINES_JSON) as f:
            data_b = json.load(f)
        records.extend(list(_iter_runs_baseline(data_b)))
    else:
        print(f"[!] {BASELINES_JSON} не найден")

    if Path(MY_JSON).exists():
        with open(MY_JSON) as f:
            data_m = json.load(f)
        records.extend(list(_iter_runs_my(data_m)))
    else:
        print(f"[!] {MY_JSON} не найден")

    records = [r for r in records if "raw_times_s" in r["run"]
                                 and len(r["run"]["raw_times_s"]) >= 20]
    return records


def rtf_array(run):
    t = np.asarray(run["raw_times_s"])
    chunk_dur_s = run["chunk_size_ms_actual"] / 1000.0
    return t / chunk_dur_s


def mean_ci_t(rtf, conf=0.95):
    n = len(rtf)
    m = rtf.mean()
    sem = rtf.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n - 1)
    half = t_crit * sem
    return {
        "n": int(n),
        "mean": float(m),
        "std": float(rtf.std(ddof=1)),
        "sem": float(sem),
        "ci_low": float(m - half),
        "ci_high": float(m + half),
        "ci_halfwidth_rel_pct": float(100 * half / m) if m > 0 else None,
    }


def quantile_ci_bootstrap(rtf, q=0.95, conf=0.95, n_boot=10_000):
    n = len(rtf)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = np.quantile(RNG.choice(rtf, size=n, replace=True), q)
    alpha = (1 - conf) / 2
    lo, hi = np.quantile(boot, [alpha, 1 - alpha])
    return {
        "q": q,
        "estimate": float(np.quantile(rtf, q)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def welch_ttest(rtf_a, rtf_b, name_a, name_b):
    t_stat, p_two = stats.ttest_ind(rtf_a, rtf_b, equal_var=False)
    p_a_less = p_two / 2 if t_stat < 0 else 1 - p_two / 2

    n_a, n_b = len(rtf_a), len(rtf_b)
    va, vb = rtf_a.var(ddof=1), rtf_b.var(ddof=1)
    s_pooled = np.sqrt(((n_a - 1) * va + (n_b - 1) * vb) / (n_a + n_b - 2))
    d = (rtf_a.mean() - rtf_b.mean()) / s_pooled if s_pooled > 0 else 0.0

    se_diff = np.sqrt(va / n_a + vb / n_b)
    df = (va/n_a + vb/n_b)**2 / ((va/n_a)**2/(n_a-1) + (vb/n_b)**2/(n_b-1))
    t_crit = stats.t.ppf(0.975, df=df)
    diff = rtf_a.mean() - rtf_b.mean()

    return {
        "model_a": name_a, "model_b": name_b,
        "n_a": n_a, "n_b": n_b,
        "mean_a": float(rtf_a.mean()), "mean_b": float(rtf_b.mean()),
        "diff_a_minus_b": float(diff),
        "diff_ci_low":  float(diff - t_crit * se_diff),
        "diff_ci_high": float(diff + t_crit * se_diff),
        "t_statistic": float(t_stat),
        "df": float(df),
        "p_value_two_sided": float(p_two),
        "p_value_a_faster_than_b": float(p_a_less),
        "cohens_d": float(d),
        "speedup_b_over_a": float(rtf_b.mean() / rtf_a.mean()) if rtf_a.mean() > 0 else None,
    }


def pick_best_cpu(records, model, chunk_ms):
    cands = [r for r in records
             if r["model"] == model and r["device"] == "cpu"
             and r["chunk_ms"] == chunk_ms]
    if not cands:
        return None
    best = min(cands, key=lambda r: np.quantile(rtf_array(r["run"]), 0.99))
    return best


def pick_gpu(records, model, chunk_ms):
    cands = [r for r in records
             if r["model"] == model and r["device"] == "gpu"
             and r["chunk_ms"] == chunk_ms]
    return cands[0] if cands else None


def main():
    records = load_all_runs()
    print(f"Loaded {len(records)} runs total")

    all_models = sorted({r["model"] for r in records})
    print("Models found:", all_models)

    ci_table = []  
    ci_json  = {}

    for model in all_models:
        ci_json[model] = {}
        for chunk_ms in CHUNKS_MS:
            for dev_key, picker in [("gpu", pick_gpu), ("cpu_best", pick_best_cpu)]:
                r = picker(records, model, chunk_ms)
                if r is None:
                    continue
                rtf = rtf_array(r["run"])
                entry = {
                    "device": dev_key,
                    "threads": r["threads"],
                    "chunk_ms": chunk_ms,
                    "rtf_mean_ci95": mean_ci_t(rtf),
                    "rtf_p95_ci95":  quantile_ci_bootstrap(rtf, q=0.95, n_boot=BOOTSTRAP_N),
                    "rtf_p99_ci95":  quantile_ci_bootstrap(rtf, q=0.99, n_boot=BOOTSTRAP_N),
                }
                ci_json[model].setdefault(dev_key, {})[f"{chunk_ms}ms"] = entry
                ci_table.append((model, dev_key, r["threads"], chunk_ms, entry))

    pairwise = []
    for ours in MY_MODEL_NAMES:
        for base in BASELINE_NAMES:
            for chunk_ms in CHUNKS_MS:
                for dev_key, picker in [("gpu", pick_gpu), ("cpu_best", pick_best_cpu)]:
                    ra = picker(records, ours, chunk_ms)
                    rb = picker(records, base, chunk_ms)
                    if ra is None or rb is None:
                        continue
                    res = welch_ttest(
                        rtf_array(ra["run"]), rtf_array(rb["run"]),
                        ours, base
                    )
                    res["device"]   = dev_key
                    res["chunk_ms"] = chunk_ms
                    res["threads_a"] = ra["threads"]
                    res["threads_b"] = rb["threads"]
                    pairwise.append(res)

    print("95% CI ДЛЯ RTF mean (t-Стьюдента) и для p95/p99 (bootstrap)")
    print(f"{'Model':<22} {'Dev':<10} {'Thr':<10} {'Chunk':<8} {'n':<5} "
          f"{'mean':<9} {'CI mean':<22} {'±%':<7} "
          f"{'p95':<9} {'CI p95':<22} {'p99':<9} {'CI p99':<22}")
    for model, dev, thr, chunk_ms, e in ci_table:
        m = e["rtf_mean_ci95"]; q95 = e["rtf_p95_ci95"]; q99 = e["rtf_p99_ci95"]
        thr_s = str(thr) if thr is not None else "-"
        ci_m = f"[{m['ci_low']:.4f}, {m['ci_high']:.4f}]"
        ci_95 = f"[{q95['ci_low']:.4f}, {q95['ci_high']:.4f}]"
        ci_99 = f"[{q99['ci_low']:.4f}, {q99['ci_high']:.4f}]"
        print(f"{model:<22} {dev:<10} {thr_s:<10} {chunk_ms}ms{'':<3} {m['n']:<5} "
              f"{m['mean']:<9.4f} {ci_m:<22} ±{m['ci_halfwidth_rel_pct']:<5.2f}% "
              f"{q95['estimate']:<9.4f} {ci_95:<22} "
              f"{q99['estimate']:<9.4f} {ci_99:<22}")

    print("Welch's t-test: H1 — НАША модель быстрее бейзлайна (одностор. p-value)")
    print(f"{'Ours':<22} {'Baseline':<12} {'Dev':<10} {'Chunk':<7} "
          f"{'mean_ours':<11} {'mean_base':<11} {'speedup':<10} {'Δ':<11} "
          f"{'t':<9} {'p (1-sided)':<13} {'d':<8} {'sig':<5}")
    for r in pairwise:
        p = r["p_value_a_faster_than_b"]
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        sp = r["speedup_b_over_a"]
        sp_s = f"{sp:.2f}x" if sp else "—"
        print(f"{r['model_a']:<22} {r['model_b']:<12} {r['device']:<10} {r['chunk_ms']}ms{'':<2} "
              f"{r['mean_a']:<11.4f} {r['mean_b']:<11.4f} {sp_s:<10} {r['diff_a_minus_b']:<+11.4f} "
              f"{r['t_statistic']:<9.2f} {p:<13.2e} {r['cohens_d']:<+8.2f} {sig:<5}")

    out = {"ci": ci_json, "pairwise_welch": pairwise}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nСохранено в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()