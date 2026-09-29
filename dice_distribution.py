"""Распределение Dice по выборке для каждого сосуда (LAD/LCx/RCA).

Одна фигура с тремя подграфиками (по одному на артерию): по X — dice, по Y —
частота. Вертикальные линии — 5/50/95 перцентили. Нужен только файл метрик.

Выход:
    out/viz/statistic/dice_distribution.png
    out/viz/statistic/dice_percentiles.csv

Запуск:
    python dice_distribution.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import common

VESSELS = ("lad", "lcx", "rca")
PERCENTILES = (5, 50, 95)
COLORS = {"lad": "#C0392B", "lcx": "#2471C4", "rca": "#1E8449"}


def load_dice_by_vessel(path: Path) -> dict:
    out = {v: [] for v in VESSELS}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "done":
                continue
            v = r.get("vessel")
            if v in out and r.get("dice"):
                out[v].append(float(r["dice"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Распределение Dice по сосудам")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    src = (Path(args.metrics) if args.metrics
           else common.OUT / "reports" / "metrics_union.csv")
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "statistic")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dice_by_vessel(src)
    rows_csv = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130,
                             layout="constrained")
    for ax, v in zip(axes, VESSELS):
        d = np.array(data[v], dtype=float)
        color = COLORS[v]
        ax.hist(d, bins="auto", range=(0.0, 1.0), color=color, alpha=0.75,
                edgecolor="black", linewidth=0.6)
        if d.size:
            p = {q: float(np.percentile(d, q)) for q in PERCENTILES}
            for q in PERCENTILES:
                ax.axvline(p[q], color="black", ls="--", lw=1.2)
                ax.text(p[q], ax.get_ylim()[1] * 0.96, f"p{q}={p[q]:.2f}",
                        rotation=90, va="top", ha="right", fontsize=8)
            ax.axvline(d.mean(), color="gray", ls=":", lw=1.4,
                       label=f"mean={d.mean():.3f}")
            ax.legend(fontsize=8, loc="upper left")
            rows_csv.append({"vessel": v.upper(), "n": int(d.size),
                             "p5": round(p[5], 4), "p50": round(p[50], 4),
                             "p95": round(p[95], 4),
                             "mean": round(float(d.mean()), 4),
                             "min": round(float(d.min()), 4),
                             "max": round(float(d.max()), 4)})
        ax.set_title(f"{v.upper()}  (n={d.size})", fontsize=12)
        ax.set_xlabel("dice")
        ax.set_ylabel("частота")
        ax.set_xlim(0.0, 1.0)
        ax.grid(alpha=0.2)

    fig.suptitle("Распределение Dice: SeqSeg (union, кроп) vs истинный фрагмент",
                 fontsize=14)
    png = out_dir / "dice_distribution.png"
    fig.savefig(png)
    plt.close(fig)

    csv_path = out_dir / "dice_percentiles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vessel", "n", "p5", "p50", "p95",
                                          "mean", "min", "max"])
        w.writeheader()
        w.writerows(rows_csv)

    print(f"сохранено: {png}\n           {csv_path}")
    for r in rows_csv:
        print(f"  {r['vessel']}: n={r['n']} p5={r['p5']} p50={r['p50']} "
              f"p95={r['p95']} mean={r['mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
