"""Примеры-визуализации по перцентилям Dice (p5/p50/p95) для трёх сосудов.

Схема — как в `viz_diagnose.py`: категориальный clscat, ровно один класс на
пиксель (приоритет пересечение > pred-only > GT-only), ровно 3 цвета:
    зелёный  — пересечение GT и предсказания;
    красный  — предсказание без GT (FP);
    синий    — GT без предсказания (FN).

Для каждой артерии берётся ближайший по dice кейс к перцентилю 5/50/95
(метрики — из out/reports/metrics_union.csv), маска — union из
out/eval/<case>/union.nii.gz.

Выход (по 3 фигуры на артерию, каждая в 3 проекциях):
    out/viz/statistic/<vessel>_p05.png, _p50.png, _p95.png

Запуск:
    python viz_percentiles.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
from common import (PROJECTION_PANELS, grey_from_hu, view_limits,
                    voxel_in_view)
from viz_diagnose import (panel_clscat, GT_COLOR, PRED_COLOR, OVER_COLOR,
                          CLICK_START, CLICK_END, CENTERLINE)

VESSELS = ("lad", "lcx", "rca")
PERCENTILES = ((5, "p05"), (50, "p50"), (95, "p95"))


def select_cases(metrics_path: Path, metric: str = "dice") -> dict:
    """Для каждой артерии — ближайший к p5/p50/p95 кейс: {vessel: {p: (case,val)}}."""
    per = {v: [] for v in VESSELS}
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "done" and r.get("vessel") in per \
                    and r.get(metric) not in (None, "", "None"):
                per[r["vessel"]].append((f"{r['scan']}_{r['vessel']}",
                                         float(r[metric])))
    selected = {}
    for v, items in per.items():
        if not items:
            continue
        vals = np.array([d for _, d in items])
        selected[v] = {}
        for q, label in PERCENTILES:
            pv = float(np.percentile(vals, q))
            case, dd = min(items, key=lambda t: abs(t[1] - pv))
            selected[v][label] = (case, dd, pv)
    return selected


def render_example(case: str, vessel: str, label: str, q: int, mval: float,
                   pv: float, out_path: Path, margin_mm: float = 20.0,
                   full: bool = False, centerline: bool = True,
                   metric: str = "dice") -> bool:
    scan = case.rsplit("_", 1)[0]
    pred_path = common.OUT / "eval" / case / "union.nii.gz"
    if not pred_path.is_file():
        print(f"  нет маски {pred_path} — пропускаю {case}")
        return False

    sample = common.load_sample(scan, vessel)
    ct, affine = common.load_ct(scan)
    gt, _ = common.load_mask(common.fragment_path(scan, vessel))
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
    grey = grey_from_hu(ct)
    view = view_limits(gt, affine, margin_mm=margin_mm)

    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)
    cl_world = np.array([c["xyz"] for c in sample["centerline"]], float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=200,
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
        ax = axes[0, ci]
        rgb, fr, fc = panel_clscat(grey, gt, pred, affine, drop, row, col)
        sr = float(abs(affine[row, row]))
        sc = float(abs(affine[col, col]))
        ax.imshow(rgb, origin="lower", interpolation="nearest", aspect=sr / sc)
        if not full:
            ax.set_xlim(*view[col])
            ax.set_ylim(*view[row])
        ax.set_xlabel(f"{clab}, индекс", fontsize=10)
        ax.set_ylabel(f"{rlab}, индекс", fontsize=10)
        ax.tick_params(labelsize=8)
        shape = rgb.shape[:2]
        if centerline:
            lc, lr = voxel_in_view(cl_world, affine, row, col, shape, fr, fc)
            ax.plot(lc, lr, "-", color=CENTERLINE, lw=2.2, alpha=1.0, zorder=8,
                    path_effects=[pe.withStroke(linewidth=3.6,
                                                foreground="black")])
        cc, rr = voxel_in_view(clicks, affine, row, col, shape, fr, fc)
        ax.plot(cc[0], rr[0], "X", color=CLICK_START, ms=10, mec="black",
                mew=1.0, zorder=7)
        ax.plot(cc[1], rr[1], "X", color=CLICK_END, ms=10, mec="black",
                mew=1.0, zorder=7)
        ax.set_title(title, fontsize=13)

    handles = [Patch(facecolor=GT_COLOR, edgecolor="black",
                     label="GT-only (синий)"),
               Patch(facecolor=PRED_COLOR, edgecolor="black",
                     label="pred-only (красный)"),
               Patch(facecolor=OVER_COLOR, edgecolor="black",
                     label="пересечение (зелёный)"),
               Line2D([0], [0], marker="X", color="none", mec="black",
                      mfc=CLICK_START, ls="none", ms=9, label="клик старта"),
               Line2D([0], [0], marker="X", color="none", mec="black",
                      mfc=CLICK_END, ls="none", ms=9, label="клик конца")]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=10)
    unit = ", мм" if "hd" in metric else ""
    fig.suptitle(
        f"{vessel.upper()} · click-crop · {label} (p{q}={pv:.3f}{unit}) · "
        f"{case} · {metric}={mval:.3f}", fontsize=15)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  сохранено: {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Примеры по перцентилям Dice")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    ap.add_argument("--metric", default="dice",
                    help="метрика для выбора кейсов (колонка CSV), "
                         "например centerline_hd95")
    ap.add_argument("--vessels", default="lad,lcx,rca",
                    help="какие артерии (через запятую)")
    ap.add_argument("--percentiles", default="5,50,95",
                    help="какие перцентили (через запятую)")
    ap.add_argument("--full", action="store_true",
                    help="полное поле (всё сердце), без кропа по GT")
    ap.add_argument("--no-centerline", dest="centerline", action="store_false",
                    help="не рисовать центрлинию GT")
    args = ap.parse_args()

    metrics_path = (Path(args.metrics) if args.metrics
                    else common.OUT / "reports" / "metrics_union.csv")
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "statistic")
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_v = {v.strip().lower() for v in args.vessels.split(",") if v.strip()}
    keep_q = {int(x) for x in args.percentiles.split(",") if x.strip()}
    percentiles = [(q, lab) for q, lab in PERCENTILES if q in keep_q]

    selected = select_cases(metrics_path, args.metric)
    n_made = 0
    for vessel in VESSELS:
        if vessel not in selected or vessel not in keep_v:
            continue
        for q, label in percentiles:
            case, mval, pv = selected[vessel][label]
            out_path = out_dir / f"{vessel}_{label}.png"
            if render_example(case, vessel, label, q, mval, pv, out_path,
                              margin_mm=args.margin_mm, full=args.full,
                              centerline=args.centerline, metric=args.metric):
                n_made += 1
    print(f"\nготово: {n_made} фигур в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
