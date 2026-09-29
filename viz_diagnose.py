"""Диагностическая визуализация, отдельно от основной (ничего не перезаписывает).

Показывает, реально ли маски «перемешаны» в 3D или это артефакт MIP:

  1) *_slab.png    — slab-MIP: проекция ограничена bbox истинного фрагмента
                     (плюс запас), поэтому далёкие структуры и перелёт в чужие
                     ветви не наслаиваются; видно честное совпадение сосуда.
  2) *_outline.png — full MIP, но GT и pred нарисованы КОНТУРАМИ поверх серого
                     КТ (без заливок), чтобы границы не «спорили» цветом.

Цвета slab: зелёный = TP, синий = FN, красный = FP.
Выход: out/viz/diagnose/<scan>_<vessel>_slab.png и _outline.png

Запуск:
    python viz_diagnose.py --scan 129 --vessel lad
    python viz_diagnose.py --scan 129 --vessel lad --scan 100 --vessel lcx
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
from scipy import ndimage as ndi

import common
from common import (PROJECTION_PANELS, grey_from_hu, orient_mip,
                    view_limits, voxel_in_view)

TP_COLOR = np.array([0.10, 0.85, 0.20])
FN_COLOR = np.array([0.20, 0.45, 0.95])
FP_COLOR = np.array([0.95, 0.15, 0.15])
GT_EDGE = np.array([0.10, 0.85, 0.20])
PRED_EDGE = np.array([0.95, 0.15, 0.15])
CLICK_START = "#FF2D95"
CLICK_END = "#00E5FF"
ALPHA = 0.85


def _slice_along(volume, axis, window):
    sl = [slice(None)] * 3
    sl[axis] = window
    return volume[tuple(sl)]


def panel_tpfn(grey, gt, pred, affine, drop, row, col, depth_slice=None):
    if depth_slice is not None:
        grey = _slice_along(grey, drop, depth_slice)
        gt = _slice_along(gt, drop, depth_slice)
        pred = _slice_along(pred, drop, depth_slice)
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    tp, _, _ = orient_mip((gt & pred).astype(np.float32), drop, row, col, affine)
    fn, _, _ = orient_mip((gt & ~pred).astype(np.float32), drop, row, col, affine)
    fp, _, _ = orient_mip((pred & ~gt).astype(np.float32), drop, row, col, affine)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    for m, c in ((tp > 0.5, TP_COLOR), (fn > 0.5, FN_COLOR), (fp > 0.5, FP_COLOR)):
        if m.any():
            rgb[m] = (1 - ALPHA) * rgb[m] + ALPHA * c
    return np.clip(rgb, 0, 1), fr, fc


def panel_channels(grey, gt, pred, affine, drop, row, col,
                   depth_slice=None):
    """Канальное наложение: GT-only синий, pred-only жёлтый, пересечение зелёное.

    R = pred & ~gt, G = pred, B = gt & ~pred  ->
      gt only  -> (0,0,1) синий
      pred only-> (1,1,0) жёлтый
      оба      -> (0,1,0) зелёный
    """
    if depth_slice is not None:
        grey = _slice_along(grey, drop, depth_slice)
        gt = _slice_along(gt, drop, depth_slice)
        pred = _slice_along(pred, drop, depth_slice)
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    r, _, _ = orient_mip((pred & ~gt).astype(np.float32), drop, row, col, affine)
    g, _, _ = orient_mip(pred.astype(np.float32), drop, row, col, affine)
    b, _, _ = orient_mip((gt & ~pred).astype(np.float32), drop, row, col, affine)
    r, g, b = r > 0.5, g > 0.5, b > 0.5
    mask_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    grey3 = np.stack([base] * 3, axis=-1).astype(np.float32)
    any_ = r | g | b
    rgb = grey3.copy()
    rgb[any_] = (1 - ALPHA) * grey3[any_] + ALPHA * mask_rgb[any_]
    return np.clip(rgb, 0, 1), fr, fc


def panel_twoc(grey, gt, pred, affine, drop, row, col, depth_slice=None):
    """Две маски по каналам и объединение: GT -> B (синий), pred -> R (красный).

    R = pred, G = 0, B = gt  ->  пересечение получается само (R+B):
      GT only  -> (0,0,1) синий
      pred only-> (1,0,0) красный
      оба      -> (1,0,1) маджента
    """
    if depth_slice is not None:
        grey = _slice_along(grey, drop, depth_slice)
        gt = _slice_along(gt, drop, depth_slice)
        pred = _slice_along(pred, drop, depth_slice)
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    r, _, _ = orient_mip(pred.astype(np.float32), drop, row, col, affine)
    b, _, _ = orient_mip(gt.astype(np.float32), drop, row, col, affine)
    r, b = r > 0.5, b > 0.5
    g = np.zeros_like(r)
    mask_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    grey3 = np.stack([base] * 3, axis=-1).astype(np.float32)
    any_ = r | b
    rgb = grey3.copy()
    rgb[any_] = (1 - ALPHA) * grey3[any_] + ALPHA * mask_rgb[any_]
    return np.clip(rgb, 0, 1), fr, fc


def panel_classes(grey, gt, pred, affine, drop, row, col, depth_slice=None):
    """Палитра A: GT-only синий, pred-only красный, пересечение зелёный.

    R = pred & ~gt, G = gt & pred, B = gt & ~pred (без жёлтого/мадженты).
    """
    if depth_slice is not None:
        grey = _slice_along(grey, drop, depth_slice)
        gt = _slice_along(gt, drop, depth_slice)
        pred = _slice_along(pred, drop, depth_slice)
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    r, _, _ = orient_mip((pred & ~gt).astype(np.float32), drop, row, col, affine)
    g, _, _ = orient_mip((gt & pred).astype(np.float32), drop, row, col, affine)
    b, _, _ = orient_mip((gt & ~pred).astype(np.float32), drop, row, col, affine)
    r, g, b = r > 0.5, g > 0.5, b > 0.5
    mask_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    grey3 = np.stack([base] * 3, axis=-1).astype(np.float32)
    any_ = r | g | b
    rgb = grey3.copy()
    rgb[any_] = (1 - ALPHA) * grey3[any_] + ALPHA * mask_rgb[any_]
    return np.clip(rgb, 0, 1), fr, fc


def panel_clscat(grey, gt, pred, affine, drop, row, col, depth_slice=None):
    """Категориально: ровно один класс на пиксель (приоритет пересечение >
    pred-only > GT-only). Итог — ровно 3 цвета, без смешений.

    зелёный = пересечение, красный = pred-only, синий = GT-only.
    """
    if depth_slice is not None:
        grey = _slice_along(grey, drop, depth_slice)
        gt = _slice_along(gt, drop, depth_slice)
        pred = _slice_along(pred, drop, depth_slice)
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    ov, _, _ = orient_mip((gt & pred).astype(np.float32), drop, row, col, affine)
    po, _, _ = orient_mip((pred & ~gt).astype(np.float32), drop, row, col, affine)
    go, _, _ = orient_mip((gt & ~pred).astype(np.float32), drop, row, col, affine)
    ov, po, go = ov > 0.5, po > 0.5, go > 0.5
    # ровно один класс на пиксель по приоритету
    g = ov
    r = po & ~ov
    b = go & ~ov & ~po
    mask_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    grey3 = np.stack([base] * 3, axis=-1).astype(np.float32)
    any_ = r | g | b
    rgb = grey3.copy()
    rgb[any_] = (1 - ALPHA) * grey3[any_] + ALPHA * mask_rgb[any_]
    return np.clip(rgb, 0, 1), fr, fc


def panel_outline(grey, gt, pred, affine, drop, row, col):
    base, fr, fc = orient_mip(grey, drop, row, col, affine)
    gt_p, _, _ = orient_mip(gt.astype(np.float32), drop, row, col, affine)
    pr_p, _, _ = orient_mip(pred.astype(np.float32), drop, row, col, affine)
    rgb = np.stack([base] * 3, axis=-1).astype(np.float32)
    st = np.ones((3, 3), bool)
    gt_e = (gt_p > 0.5) & ~ndi.binary_erosion(gt_p > 0.5, structure=st)
    pr_e = (pr_p > 0.5) & ~ndi.binary_erosion(pr_p > 0.5, structure=st)
    rgb[gt_e] = GT_EDGE
    rgb[pr_e] = PRED_EDGE
    return np.clip(rgb, 0, 1), fr, fc


def make_figures(scan, vessel, out_dir, margin_mm, slab_mm, tags=None):
    ct, affine = common.load_ct(scan)
    gt, _ = common.load_mask(common.fragment_path(scan, vessel))
    pred_path = common.OUT / "eval" / f"{scan}_{vessel}" / "union.nii.gz"
    if not pred_path.is_file():
        print(f"  нет {pred_path}")
        return 0
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
    grey = grey_from_hu(ct)
    view = view_limits(gt, affine, margin_mm=margin_mm)
    sample = common.load_sample(scan, vessel)
    cl_xyz = np.array([c["xyz"] for c in sample["centerline"]], float)
    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)

    # bbox GT вдоль каждой оси (для slab)
    idx = np.argwhere(gt)
    lo, hi = idx.min(0), idx.max(0)

    all_tags = (("slab", panel_tpfn), ("outline", panel_outline),
                ("channels", panel_channels), ("twoc", panel_twoc),
                ("twocA", panel_classes), ("clscat", panel_clscat))
    if tags:
        all_tags = tuple((t, fn) for t, fn in all_tags if t in tags)

    n_made = 0
    for tag, fn in all_tags:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=200,
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0,
                                    hspace=0.0)
        for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
            ax = axes[0, ci]
            if tag == "slab":
                sp = float(abs(affine[drop, drop]))
                m = int(np.ceil(slab_mm / sp))
                w = slice(max(0, int(lo[drop]) - m),
                          min(gt.shape[drop], int(hi[drop]) + m + 1))
                rgb, fr, fc = fn(grey, gt, pred, affine, drop, row, col,
                                 depth_slice=w)
            else:
                rgb, fr, fc = fn(grey, gt, pred, affine, drop, row, col)
            sr = float(abs(affine[row, row]))
            sc = float(abs(affine[col, col]))
            ax.imshow(rgb, origin="lower", interpolation="nearest",
                      aspect=sr / sc)
            ax.set_xlim(*view[col])
            ax.set_ylim(*view[row])
            ax.set_xlabel(f"{clab}, индекс", fontsize=10)
            ax.set_ylabel(f"{rlab}, индекс", fontsize=10)
            ax.tick_params(labelsize=8)
            if tag not in ("twoc", "twocA", "clscat"):
                c, r = voxel_in_view(cl_xyz, affine, row, col,
                                     rgb.shape[:2], fr, fc)
                ax.plot(c, r, "-", color="black", lw=1.2, zorder=6)
            cc, rr = voxel_in_view(clicks, affine, row, col, rgb.shape[:2], fr, fc)
            ax.plot(cc[0], rr[0], "X", color=CLICK_START, ms=10, mec="black",
                    mew=1.0, zorder=7)
            ax.plot(cc[1], rr[1], "X", color=CLICK_END, ms=10, mec="black",
                    mew=1.0, zorder=7)
            ax.set_title(title, fontsize=13)

        if tag == "slab":
            handles = [Patch(facecolor=TP_COLOR, edgecolor="black", label="TP"),
                       Patch(facecolor=FN_COLOR, edgecolor="black", label="FN"),
                       Patch(facecolor=FP_COLOR, edgecolor="black", label="FP")]
            note = f"slab-MIP: глубина ограничена bbox GT ± {slab_mm:.0f} мм"
        elif tag == "channels":
            handles = [Patch(facecolor=(0.10, 0.30, 0.95), edgecolor="black",
                             label="GT (синий)"),
                       Patch(facecolor=(0.98, 0.90, 0.10), edgecolor="black",
                             label="pred (жёлтый)"),
                       Patch(facecolor=(0.10, 0.85, 0.20), edgecolor="black",
                             label="пересечение (зелёный)")]
            note = "каналы: GT=синий, pred=жёлтый, пересечение=зелёный"
        elif tag == "twoc":
            handles = [Patch(facecolor=(0.10, 0.25, 0.95), edgecolor="black",
                             label="GT (канал B, синий)"),
                       Patch(facecolor=(0.92, 0.15, 0.15), edgecolor="black",
                             label="pred (канал R, красный)"),
                       Patch(facecolor=(0.90, 0.20, 0.90), edgecolor="black",
                             label="пересечение (R+B, маджента)")]
            note = "GT=B (синий), pred=R (красный), пересечение=маджента"
        elif tag == "twocA":
            handles = [Patch(facecolor=(0.10, 0.35, 0.95), edgecolor="black",
                             label="GT (синий)"),
                       Patch(facecolor=(0.90, 0.15, 0.15), edgecolor="black",
                             label="pred (красный)"),
                       Patch(facecolor=(0.10, 0.85, 0.20), edgecolor="black",
                             label="пересечение (зелёный)")]
            note = "GT=синий, pred=красный, пересечение=зелёный"
        elif tag == "clscat":
            handles = [Patch(facecolor=(0.10, 0.35, 0.95), edgecolor="black",
                             label="GT-only (синий)"),
                       Patch(facecolor=(0.90, 0.15, 0.15), edgecolor="black",
                             label="pred-only (красный)"),
                       Patch(facecolor=(0.10, 0.85, 0.20), edgecolor="black",
                             label="пересечение (зелёный)")]
            note = "категориально: 1 класс на пиксель — ровно 3 цвета"
        else:
            handles = [Patch(facecolor=GT_EDGE, edgecolor="black", label="контур GT"),
                       Patch(facecolor=PRED_EDGE, edgecolor="black", label="контур pred")]
            note = "full MIP, контуры без заливок"
        if tag not in ("twoc", "twocA", "clscat"):
            handles += [Line2D([0], [0], color="black", lw=1.2,
                               label="центрлиния")]
        handles += [Line2D([0], [0], marker="X", color="none", mec="black",
                           mfc=CLICK_START, ls="none", ms=9, label="клик старта"),
                    Line2D([0], [0], marker="X", color="none", mec="black",
                           mfc=CLICK_END, ls="none", ms=9, label="клик конца")]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles),
                   frameon=False, fontsize=10)
        fig.suptitle(f"{scan} · {vessel.upper()} · {note}", fontsize=15)
        p = out_dir / f"{scan}_{vessel}_{tag}.png"
        fig.savefig(p)
        plt.close(fig)
        print(f"  сохранено: {p}")
        n_made += 1
    return n_made


def main() -> int:
    ap = argparse.ArgumentParser(description="Диагностика масок (slab/outline)")
    ap.add_argument("--scan", action="append", required=True)
    ap.add_argument("--vessel", action="append", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    ap.add_argument("--slab-mm", type=float, default=6.0)
    ap.add_argument("--tags", nargs="+", default=None,
                    help="какие варианты делать: slab outline channels twoc twocA")
    args = ap.parse_args()

    if len(args.scan) != len(args.vessel):
        print("число --scan и --vessel должно совпадать")
        return 1
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "diagnose")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for scan, vessel in zip(args.scan, args.vessel):
        print(f"{scan}_{vessel}:")
        n += make_figures(scan, vessel, out_dir, args.margin_mm, args.slab_mm,
                          tags=args.tags)
    print(f"\nготово: {n} фигур в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
