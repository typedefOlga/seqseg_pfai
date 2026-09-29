"""Визуализация масок SeqSeg поверх истинного фрагмента в трёх проекциях.

Строки — варианты (F, B, F∪B, F∩B), столбцы — сагиттальная/корональная/
аксиальная MIP-проекции. Истинная маска — зелёная, предсказание — красное,
пересечение — жёлтое, контур GT — белый. Фон — серый MIP КТ.

Запуск:
    python visualize.py --scan 908 --vessel lcx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common
from common import PROJECTION_PANELS, grey_from_hu, orient_mip, view_limits, voxel_in_view

GT_COLOR = np.array([0.10, 0.85, 0.20])   # истинная маска — зелёный
BOTH_COLOR = np.array([0.98, 0.90, 0.10])  # пересечение GT и предикта — жёлтый
CLICK_START = "#FF2D95"
CLICK_END = "#00E5FF"
ALPHA = 0.85

VARIANT_ORDER = ["F", "B", "UNION", "INTER"]
VARIANT_TITLE = {
    "F": "F — от старта",
    "B": "B — от конца",
    "UNION": "F ∪ B",
    "INTER": "F ∩ B",
}
# Разные цвета предикта для разных вариантов: F — красный, B — синий.
VARIANT_PRED_COLOR = {
    "F": np.array([0.95, 0.15, 0.15]),
    "B": np.array([0.15, 0.45, 0.95]),
    "UNION": np.array([0.80, 0.20, 0.85]),
    "INTER": np.array([0.95, 0.55, 0.05]),
}


def load_variants(out_dir: Path) -> dict:
    variants = {}
    for name in VARIANT_ORDER:
        p = out_dir / f"{name.lower()}.nii.gz"
        if p.is_file():
            variants[name] = np.asanyarray(nib.load(str(p)).dataobj) > 0
    return variants


def composite_panel(grey, gt_m, pred_m, affine, drop, row, col,
                    pred_color=GT_COLOR):
    """RGB-панель: серый MIP + GT(зелёный)/pred(цвет варианта)/пересечение(жёлтый)."""
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    gt, _, _ = orient_mip(gt_m.astype(np.float32), drop, row, col, affine)
    pred, _, _ = orient_mip(pred_m.astype(np.float32), drop, row, col, affine)
    gt = gt > 0.5
    pred = pred > 0.5
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)

    for mask, color in ((gt & ~pred, GT_COLOR),
                        (pred & ~gt, np.asarray(pred_color, dtype=np.float32)),
                        (gt & pred, BOTH_COLOR)):
        if mask.any():
            rgb[mask] = (1 - ALPHA) * rgb[mask] + ALPHA * color
    return np.clip(rgb, 0, 1), fr, fc


def main() -> int:
    ap = argparse.ArgumentParser(description="Проекции масок SeqSeg vs GT")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--vessel", required=True)
    ap.add_argument("--evaldir", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    args = ap.parse_args()
    scan, vessel = args.scan, args.vessel

    out_dir = (Path(args.evaldir) if args.evaldir
               else common.OUT / "eval" / f"{scan}_{vessel.lower()}")
    variants = load_variants(out_dir)
    if not variants:
        print(f"нет масок в {out_dir} — сначала evaluate.py")
        return 1

    sample = common.load_sample(scan, vessel)
    ct, affine = common.load_ct(scan)
    gt, _ = common.load_mask(common.fragment_path(scan, vessel))
    grey = grey_from_hu(ct)
    view = view_limits(gt, affine, margin_mm=args.margin_mm)

    cl_xyz = np.array([c["xyz"] for c in sample["centerline"]], float)
    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)

    metrics = {}
    for name in variants:
        j = out_dir / f"{name.lower()}.json"
        if j.is_file():
            import json
            metrics[name] = json.loads(j.read_text(encoding="utf-8"))

    viz_dir = (Path(args.outdir) if args.outdir else common.OUT / "viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    names = [n for n in VARIANT_ORDER if n in variants]
    fig, axes = plt.subplots(len(names), 3, figsize=(16, 4.6 * len(names)),
                             dpi=130, layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    for ri, name in enumerate(names):
        pred_m = variants[name]
        for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
            ax = axes[ri, ci]
            panel, fr, fc = composite_panel(
                grey, gt, pred_m, affine, drop, row, col,
                VARIANT_PRED_COLOR.get(name, (0.95, 0.15, 0.15)))
            sr, sc = float(abs(affine[row, row])), float(abs(affine[col, col]))
            ax.imshow(panel, origin="lower", interpolation="nearest",
                      aspect=sr / sc)
            ax.set_xlim(*view[col])
            ax.set_ylim(*view[row])
            ax.set_xlabel(f"{clab}, индекс", fontsize=9)
            ax.set_ylabel(f"{rlab}, индекс", fontsize=9)
            ax.tick_params(labelsize=7)

            c, r = voxel_in_view(cl_xyz, affine, row, col, panel.shape[:2], fr, fc)
            ax.plot(c, r, "-", color="black", lw=1.3, zorder=6)
            cc, rr = voxel_in_view(clicks, affine, row, col, panel.shape[:2], fr, fc)
            ax.plot(cc[0], rr[0], "X", color=CLICK_START, ms=11, mec="black",
                    mew=1.0, zorder=7)
            ax.plot(cc[1], rr[1], "X", color=CLICK_END, ms=11, mec="black",
                    mew=1.0, zorder=7)

            ax.set_title(f"{VARIANT_TITLE.get(name, name)} · {title}",
                         fontsize=12,
                         color=VARIANT_PRED_COLOR.get(name, "black"))
            m = metrics.get(name, {})
            if m:
                ax.text(0.02, 0.98,
                        f"dice={m.get('dice', float('nan')):.3f}\n"
                        f"clDice={m.get('cl_dice', float('nan')):.3f}\n"
                        f"HD95={m.get('hd95', float('nan')):.1f} мм",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=9, color="white",
                        bbox=dict(facecolor="black", alpha=0.5, pad=2))

    handles = [
        Patch(facecolor=GT_COLOR, edgecolor="black", label="истинный фрагмент"),
        Patch(facecolor=VARIANT_PRED_COLOR["F"], edgecolor="black",
              label="F — от старта"),
        Patch(facecolor=VARIANT_PRED_COLOR["B"], edgecolor="black",
              label="B — от конца"),
        Patch(facecolor=BOTH_COLOR, edgecolor="black",
              label="пересечение GT и предикта"),
        Line2D([0], [0], color="black", lw=1.3, label="центрлиния фрагмента"),
        Line2D([0], [0], marker="X", color="none", mec="black",
               mfc=CLICK_START, ls="none", ms=10, label="клик старта"),
        Line2D([0], [0], marker="X", color="none", mec="black",
               mfc=CLICK_END, ls="none", ms=10, label="клик конца"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
               frameon=False, fontsize=10)
    fig.suptitle(
        f"Скан {scan} · {vessel.upper()} · SeqSeg vs истинный фрагмент"
        f" · длина {sample['length_mm']:.1f} мм", fontsize=15)

    out_png = viz_dir / f"{scan}_{vessel.lower()}_masks.png"
    fig.savefig(out_png)
    plt.close(fig)
    print(f"сохранено: {out_png}")

    # отдельная картинка на каждый вариант
    for name in names:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=130,
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0,
                                    hspace=0.0)
        for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
            ax = axes[0, ci]
            panel, fr, fc = composite_panel(
                grey, gt, variants[name], affine, drop, row, col,
                VARIANT_PRED_COLOR.get(name, (0.95, 0.15, 0.15)))
            sr, sc = float(abs(affine[row, row])), float(abs(affine[col, col]))
            ax.imshow(panel, origin="lower", interpolation="nearest",
                      aspect=sr / sc)
            ax.set_xlim(*view[col]); ax.set_ylim(*view[row])
            ax.set_xlabel(f"{clab}, индекс", fontsize=9)
            ax.set_ylabel(f"{rlab}, индекс", fontsize=9)
            c, r = voxel_in_view(cl_xyz, affine, row, col, panel.shape[:2], fr, fc)
            ax.plot(c, r, "-", color="black", lw=1.3, zorder=6)
            ax.set_title(title, fontsize=12)
        fig.suptitle(f"{scan} · {vessel.upper()} · {VARIANT_TITLE.get(name, name)}",
                     fontsize=14)
        p = viz_dir / f"{scan}_{vessel.lower()}_{name.lower()}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"сохранено: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
