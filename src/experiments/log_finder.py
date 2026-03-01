from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.config.paths import ARTIFACTS_DIR


def safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def find_best(jsonl_path: Path) -> Tuple[Optional[Dict[str, Any]], int, int]:
    best_row = None
    best_score = None
    n_total = 0
    n_scored = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            metrics = row.get("metrics", {}) or {}
            score = safe_float(metrics.get("mean_log_loss"))
            if score is None:
                continue

            n_scored += 1
            if (best_score is None) or (score < best_score):
                best_score = score
                best_row = row

    return best_row, n_total, n_scored


def main():
    jsonl_path = Path(ARTIFACTS_DIR, "experiments/experiments.jsonl")

    best, n_total, n_scored = find_best(jsonl_path)

    print(f"[info] total_lines={n_total}, scored_lines={n_scored}")
    if best is None:
        print("[result] No valid mean_log_loss found.")
        return

    metrics = best.get("metrics", {}) or {}
    params = best.get("params", {}) or {}

    print("\n=== BEST EXPERIMENT (min mean_log_loss) ===")
    print(f"run_id: {best.get('run_id')}")
    print(f"timestamp: {best.get('timestamp')}")
    print(f"tag: {best.get('tag')}")
    print(f"mean_log_loss: {metrics.get('mean_log_loss')}")
    print(f"std_log_loss:  {metrics.get('std_log_loss')}")
    print(f"fold_log_loss: {metrics.get('fold_log_loss')}")
    print("\n-- params.logreg --")
    print(params.get("logreg"))
    print("\n-- params.hog --")
    print(params.get("hog"))
    print("-"*20)
    print(json.dumps(best, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()