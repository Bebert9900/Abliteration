"""Compare base vs abliteré sur MMLU + GSM8K via le câblage lm-eval du repo (abliteration.eval.run_benchmark).

Usage : python run_benchmarks.py <base_model> <ablated_dir> [--limit N] [--tasks mmlu,gsm8k]
Écrit benchmarks_compare.json. `--limit` restreint le nombre d'exemples (run rapide).
"""
import argparse
import json
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root sur sys.path

from abliteration.eval import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("ablated")
    ap.add_argument("--tasks", default="mmlu,gsm8k")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ns = ap.parse_args()

    tasks = [t.strip() for t in ns.tasks.split(",") if t.strip()]
    out = {"base": ns.base, "ablated": ns.ablated, "limit": ns.limit, "results": {}}
    for label, model in [("base", ns.base), ("ablated", ns.ablated)]:
        out["results"][label] = {}
        for task in tasks:
            print(f"[bench] {label} / {task} …", flush=True)
            res = run_benchmark(task, model, device=ns.device, limit=ns.limit)
            out["results"][label][task] = res
            print(f"[bench] {label} / {task} = {res['score']:.4f} ({res['metric']})", flush=True)
            with open("benchmarks_compare.json", "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
