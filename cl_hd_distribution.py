"""Распределение centerline_hd95 (cl_hd95) по выборке для каждого сосуда.

Отдельный скрипт (аналог dice_distribution.py) для метрики cl_hd95 по
click-crop-маскам. Одна фигура с тремя подграфиками (LAD/LCx/RCA): по X —
cl_hd95 (мм), по Y — частота; вертикальные линии — 5/50/95 перцентили и mean.
Диапазон по X — полный (ничего не обрезается). Нужен только файл метрик.

Выход:
    out/viz/statistic_crop/cl_hd95_distribution.png
    out/viz/statistic_crop/cl_hd95_percentiles.csv

Запуск:
    SEQSEG_OUT=out0610 python cl_hd_distribution.py
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

METRIC = "centerline_hd95"
VESSELS = ("lad", "lcx", "rca")
PERCENTILES = (5, 50, 95)
COLORS = {"lad": "#C0392B", "lcx": "#2471C4", "rca": "#1E8449"}


def load_metric_by_vessel(path: Path) -> dict:
    out = {v: [] for v in VESSELS}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "done":
                continue
            v = r.get("vessel")
            if v in out and r.get(METRIC) not in (None, "", "None"):
                out[v].append(float(r[METRIC]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Распределение cl_hd95 по сосудам")
    ap.add_argument("--metrics", default=None,
                    help=f"файл метрик (по умолчанию click-crop "
                         f"{common.OUT / 'recount' / 'run120' / 'metrics_cropped.csv'})")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--title",
                    default="Распределение cl_hd95: SeqSeg (click-crop) "
                            "vs истинный фрагмент")
    args = ap.parse_args()

    src = (Path(args.metrics) if args.metrics
           else common.OUT / "recount" / "run120" / "metrics_cropped.csv")
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "statistic_crop")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_metric_by_vessel(src)
    rows_csv = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130,
                             layout="constrained")
    for ax, v in zip(axes, VESSELS):
        d = np.array(data[v], dtype=float)
        color = COLORS[v]
        xmax = float(d.max()) if d.size else 1.0
        xmax = xmax * 1.02 if xmax > 0 else 1.0
        ax.hist(d, bins="auto", range=(0.0, xmax), color=color, alpha=0.75,
                edgecolor="black", linewidth=0.6)
        if d.size:
            p = {q: float(np.percentile(d, q)) for q in PERCENTILES}
            for q in PERCENTILES:
                ax.axvline(p[q], color="black", ls="--", lw=1.2)
                ax.text(p[q], ax.get_ylim()[1] * 0.96, f"p{q}={p[q]:.2f}",
                        rotation=90, va="top", ha="right", fontsize=8)
            ax.axvline(d.mean(), color="gray", ls=":", lw=1.4,
                       label=f"mean={d.mean():.2f}")
            ax.legend(fontsize=8, loc="upper right")
            rows_csv.append({"vessel": v.upper(), "n": int(d.size),
                             "p5": round(p[5], 4), "p50": round(p[50], 4),
                             "p95": round(p[95], 4),
                             "mean": round(float(d.mean()), 4),
                             "min": round(float(d.min()), 4),
                             "max": round(float(d.max()), 4)})
        ax.set_title(f"{v.upper()}  (n={d.size})", fontsize=12)
        ax.set_xlabel("cl_hd95, мм")
        ax.set_ylabel("частота")
        ax.set_xlim(0.0, xmax)
        ax.grid(alpha=0.2)

    fig.suptitle(args.title, fontsize=14)
    png = out_dir / "cl_hd95_distribution.png"
    fig.savefig(png)
    plt.close(fig)

    csv_path = out_dir / "cl_hd95_percentiles.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vessel", "n", "p5", "p50", "p95",
                                          "mean", "min", "max"])
        w.writeheader()
        w.writerows(rows_csv)

    print(f"сохранено: {png}\n           {csv_path}")
    for r in rows_csv:
        print(f"  {r['vessel']}: n={r['n']} p5={r['p5']} p50={r['p50']} "
              f"p95={r['p95']} mean={r['mean']} max={r['max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
