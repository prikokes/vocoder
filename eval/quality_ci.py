import json
import numpy as np
from scipy import stats
from pathlib import Path

INPUT_JSONS = [
    "benchmark_results.json",
    "benchmark_results_repro.json",
    "benchmark_results_my.json",
]
OUTPUT_JSON = "quality_ci.json"

CONFIDENCE  = 0.95
BOOTSTRAP_N = 10_000
RNG         = np.random.default_rng(42)

METRICS = ["pesq", "stoi", "lsd", "mel_distance"]
METRIC_DIRECTION = {
    "pesq": "↑", "stoi": "↑", "lsd": "↓", "mel_distance": "↓",
}


def mean_ci_t(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return None
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n - 1)
    half = t_crit * sem
    return {
        "method": "t-Student",
        "n": int(n),
        "mean": float(m),
        "std": float(x.std(ddof=1)),
        "sem": float(sem),
        "ci_low":  float(m - half),
        "ci_high": float(m + half),
        "ci_halfwidth": float(half),
        "ci_halfwidth_rel_pct": float(100 * half / m) if m != 0 else None,
    }


def mean_ci_bootstrap(x, conf=0.95, n_boot=10_000):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return None
    m = x.mean()
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        boot_means[i] = RNG.choice(x, size=n, replace=True).mean()
    alpha = (1 - conf) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {
        "method": "bootstrap-percentile",
        "n": int(n),
        "n_boot": n_boot,
        "mean": float(m),
        "ci_low":  float(lo),
        "ci_high": float(hi),
        "ci_halfwidth": float((hi - lo) / 2),
        "ci_halfwidth_rel_pct": float(100 * (hi - lo) / 2 / m) if m != 0 else None,
    }


def load_all():
    merged = {}
    for path in INPUT_JSONS:
        if not Path(path).exists():
            print(f"[skip] {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for model_name, content in data.items():
            if isinstance(content, dict) and "raw" in content:
                raw = content["raw"]
            elif isinstance(content, dict) and any(k in content for k in METRICS):
                print(f"[skip] {path}::{model_name} — нет сырых значений (raw), "
                      f"CI посчитать нельзя. Перезапусти бенчмарк.")
                continue
            else:
                continue
            key = model_name
            suffix = 1
            while key in merged:
                suffix += 1
                key = f"{model_name}#{suffix}"
            merged[key] = raw
    return merged


def main():
    all_raw = load_all()
    if not all_raw:
        print("Нет данных с сырыми значениями. Сначала перезапусти бенчмарк "
              "с правкой evaluate_model (см. инструкцию).")
        return

    out = {}
    print(f"\nМоделей: {len(all_raw)}")
    print(f"Bootstrap resamples: {BOOTSTRAP_N}\n")

    for model_name, raw_by_metric in all_raw.items():
        out[model_name] = {}
        for metric in METRICS:
            vals = raw_by_metric.get(metric, [])
            if not vals:
                continue
            ci_boot = mean_ci_bootstrap(vals, conf=CONFIDENCE, n_boot=BOOTSTRAP_N)
            ci_t    = mean_ci_t(vals, conf=CONFIDENCE)
            out[model_name][metric] = {
                "bootstrap": ci_boot,
                "t_student": ci_t,
            }

    print("95% Доверительные интервалы для метрик качества")
    print("Bootstrap (10k resamples, percentile). В скобках — t-интервал Стьюдента для сравнения.")

    header = f"{'Model':<22}"
    for m in METRICS:
        header += f"{m.upper()+' '+METRIC_DIRECTION[m]:<28}"
    print(header)
    print("-" * 130)

    for model_name, by_metric in out.items():
        row = f"{model_name:<22}"
        for m in METRICS:
            entry = by_metric.get(m)
            if entry is None:
                row += f"{'—':<28}"
                continue
            b = entry["bootstrap"]
            cell = f"{b['mean']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
            row += f"{cell:<28}"
        print(row)
    print()

    print("Детально (n, mean, std, bootstrap CI, t-CI, относительная полуширина)")
    for model_name, by_metric in out.items():
        print(f"\n[{model_name}]")
        for m in METRICS:
            entry = by_metric.get(m)
            if entry is None:
                continue
            b = entry["bootstrap"]; t = entry["t_student"]
            print(f"  {m.upper():<14} n={b['n']:<4} "
                  f"mean={b['mean']:.4f}  "
                  f"BS-CI=[{b['ci_low']:.4f}, {b['ci_high']:.4f}] "
                  f"(±{b['ci_halfwidth']:.4f}, ±{b['ci_halfwidth_rel_pct']:.2f}%)  "
                  f"t-CI=[{t['ci_low']:.4f}, {t['ci_high']:.4f}]  "
                  f"std={t['std']:.4f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nСохранено в {OUTPUT_JSON}")


if __name__ == "__main__":
    main()