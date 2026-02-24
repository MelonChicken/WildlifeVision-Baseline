from __future__ import annotations

import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from src.config.paths import PROJECT_ROOT, LOG_PATH

OUT_DIR = PROJECT_ROOT / "docs/experiments/plot"
OUT_PNG = OUT_DIR / "experiment_progress.png"
OUT_SVG = OUT_DIR / "experiment_progress.svg"


def _get(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _parse_ts(ts: str | None):
    if not ts:
        return None
    try:
        # handles "2026-02-14T07:28:01.276826+00:00" and "Z"
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Missing: {LOG_PATH}")

    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            ts = _parse_ts(_get(rec, "timestamp"))
            mean = _get(rec, "metrics.mean_log_loss")
            if ts is None or mean is None:
                continue

            rows.append(
                {
                    "ts": ts,
                    "mean": float(mean),
                    "tag": _get(rec, "tag", "-"),
                    "C": _get(rec, "params.logreg.C", None),
                    "use_scaler": _get(rec, "params.logreg.use_scaler", None),
                    "class_weight": _get(rec, "params.logreg.class_weight", None),
                    "max_iter": _get(rec, "params.logreg.max_iter", None),
                    "n_splits": _get(rec, "cv.n_splits", None),
                }
            )

    if not rows:
        raise RuntimeError("No valid rows with timestamp + mean_log_loss found.")

    rows.sort(key=lambda r: r["ts"])
    t = [r["ts"] for r in rows]
    run_idx = np.arange(1, len(rows) + 1, dtype=int)
    y = np.array([r["mean"] for r in rows], dtype=float)

    # best-so-far + best point
    best = []
    cur = float("inf")
    best_i = 0
    best_points = []
    for i, v in enumerate(y):
        if v < cur:
            cur = v
            best_i = i
            best_points.append(i)
        best.append(cur)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(12.0, 5.6))

    ax.scatter(t, y, s=18, alpha=0.45, label="Run")
    ax.plot(t, best, linestyle="--", linewidth=1.6, label="Best so far")

    # Mark every new best point to show performance jumps.
    bt_series = [run_idx[i] for i in best_points]
    by_series = [y[i] for i in best_points]
    ax.scatter(bt_series, by_series, s=34, zorder=3, label="New best")

    bt, by = int(run_idx[best_i]), y[best_i]
    ax.scatter([bt], [by], s=56, zorder=4)
    ax.annotate(
        f"global best {by:.4f}",
        xy=(bt, by),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=9,
        ha="left",
        va="bottom",
    )

    latest_date = rows[-1]["ts"].date().isoformat()

    if len(y) > 1:
        first = float(y[0])
        gain = first - float(np.min(y))
        ax.text(
            0.01,
            0.98,
            f"Runs: {len(y)} | Latest: {latest_date} | First: {first:.4f} | "
            f"Best: {np.min(y):.4f} | Gain: {gain:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.12, lw=0),
        )
    else:
        ax.text(
            0.01,
            0.98,
            f"Runs: {len(y)} | Latest: {latest_date} | Best: {np.min(y):.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.12, lw=0),
        )

    ax.set_title("Experiment Progress")
    ax.set_xlabel("Run #")
    ax.set_ylabel("CV mean log loss (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_xlim(1, len(rows))
    step = max(1, len(rows) // 10)
    ax.set_xticks(np.arange(1, len(rows) + 1, step))

    fig.tight_layout()
    fig.savefig(OUT_PNG)
    fig.savefig(OUT_SVG)
    plt.close(fig)

    print(f"[ok] saved -> {OUT_PNG}")
    print(f"[ok] saved -> {OUT_SVG}")


if __name__ == "__main__":
    main()
