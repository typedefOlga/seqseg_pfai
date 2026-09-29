"""Примеры-визуализации по перцентилям Dice (p5/p50/p95) для трёх сосудов.

Основа — `visualize.py`, но цветовая схема TP/FN/FP:
    зелёный  — совпадение GT и предсказания (TP);
    синий    — GT без предсказания (FN);
    красный  — предсказание без GT (FP).
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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
from common import (PROJECTION_PANELS, grey_from_hu, orient_mip,
                    view_limits, voxel_in_view)

TP_COLOR = np.array([0.10, 0.85, 0.20])   # GT ∩ pred (совпадение)
FN_COLOR = np.array([0.20, 0.45, 0.95])   # GT \ pred (пропуск)
FP_COLOR = np.array([0.95, 0.15, 0.15])   # pred \ GT (лишнее)
CLICK_START = "#FF2D95"
CLICK_END = "#00E5FF"
ALPHA = 0.85

VESSELS = ("lad", "lcx", "rca")
PERCENTILES = ((5, "p05"), (50, "p50"), (95, "p95"))


def panel_tpfn(grey, gt_m, pred_m, affine, drop, row, col):
    """RGB-панель MIP: TP(зелёный)/FN(синий)/FP(красный) на сером КТ.

    Классы считаются в 3D и проецируются раздельно (корректно по глубине).
    """
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    tp, _, _ = orient_mip((gt_m & pred_m).astype(np.float32), drop, row, col, affine)
    fn, _, _ = orient_mip((gt_m & ~pred_m).astype(np.float32), drop, row, col, affine)
    fp, _, _ = orient_mip((pred_m & ~gt_m).astype(np.float32), drop, row, col, affine)

    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    for mask, color in ((tp > 0.5, TP_COLOR),
                        (fn > 0.5, FN_COLOR),
                        (fp > 0.5, FP_COLOR)):
        if mask.any():
            rgb[mask] = (1 - ALPHA) * rgb[mask] + ALPHA * color
    return np.clip(rgb, 0, 1), fr, fc


def select_cases(metrics_path: Path) -> dict:
    """Для каждой артерии — ближайший к p5/p50/p95 кейс: {vessel: {p: (case,dice)}}."""
    per = {v: [] for v in VESSELS}
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "done" and r.get("vessel") in per:
                per[r["vessel"]].append((f"{r['scan']}_{r['vessel']}",
                                         float(r["dice"])))
    selected = {}
    for v, items in per.items():
        if not items:
            continue
        dice = np.array([d for _, d in items])
        selected[v] = {}
        for q, label in PERCENTILES:
            pv = float(np.percentile(dice, q))
            case, dd = min(items, key=lambda t: abs(t[1] - pv))
            selected[v][label] = (case, dd, pv)
    return selected


def render_example(case: str, vessel: str, label: str, q: int, dice: float,
                   pv: float, out_path: Path, margin_mm: float = 20.0) -> bool:
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

    cl_xyz = np.array([c["xyz"] for c in sample["centerline"]], float)
    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), dpi=130,
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
        ax = axes[0, ci]
        rgb, fr, fc = panel_tpfn(grey, gt, pred, affine, drop, row, col)
        sr = float(abs(affine[row, row]))
        sc = float(abs(affine[col, col]))
        ax.imshow(rgb, origin="lower", interpolation="nearest", aspect=sr / sc)
        ax.set_xlim(*view[col])
        ax.set_ylim(*view[row])
        ax.set_xlabel(f"{clab}, индекс", fontsize=9)
        ax.set_ylabel(f"{rlab}, индекс", fontsize=9)
        ax.tick_params(labelsize=7)
        c, r = voxel_in_view(cl_xyz, affine, row, col, rgb.shape[:2], fr, fc)
        ax.plot(c, r, "-", color="black", lw=1.3, zorder=6)
        cc, rr = voxel_in_view(clicks, affine, row, col, rgb.shape[:2], fr, fc)
        ax.plot(cc[0], rr[0], "X", color=CLICK_START, ms=11, mec="black",
                mew=1.0, zorder=7)
        ax.plot(cc[1], rr[1], "X", color=CLICK_END, ms=11, mec="black",
                mew=1.0, zorder=7)
        ax.set_title(title, fontsize=12)

    handles = [
        Patch(facecolor=TP_COLOR, edgecolor="black", label="TP — совпадение GT и pred"),
        Patch(facecolor=FN_COLOR, edgecolor="black", label="FN — GT не найден"),
        Patch(facecolor=FP_COLOR, edgecolor="black", label="FP — лишнее у pred"),
        Line2D([0], [0], color="black", lw=1.3, label="центрлиния фрагмента"),
        Line2D([0], [0], marker="X", color="none", mec="black",
               mfc=CLICK_START, ls="none", ms=10, label="клик старта"),
        Line2D([0], [0], marker="X", color="none", mec="black",
               mfc=CLICK_END, ls="none", ms=10, label="клик конца"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               frameon=False, fontsize=9)
    fig.suptitle(
        f"{vessel.upper()} · {label} (p{q}={pv:.3f}) · {case} · dice={dice:.3f}",
        fontsize=15)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  сохранено: {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Примеры по перцентилям Dice")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    args = ap.parse_args()

    metrics_path = (Path(args.metrics) if args.metrics
                    else common.OUT / "reports" / "metrics_union.csv")
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "statistic")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = select_cases(metrics_path)
    n_made = 0
    for vessel in VESSELS:
        if vessel not in selected:
            continue
        for q, label in PERCENTILES:
            case, dice, pv = selected[vessel][label]
            out_path = out_dir / f"{vessel}_{label}.png"
            if render_example(case, vessel, label, q, dice, pv, out_path,
                              margin_mm=args.margin_mm):
                n_made += 1
    print(f"\nготово: {n_made} фигур в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
