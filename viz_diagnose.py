"""Визуализация масок: категориальная схема «синий / красный / зелёный».

Ровно один класс на пиксель (приоритет пересечение > pred-only > GT-only),
поэтому ровно 3 цвета без смешений:
    зелёный  — пересечение GT и предсказания;
    красный  — предсказание без GT (FP);
    синий    — GT без предсказания (FN).

Классы считаются в 3D по вокселям и лишь затем проецируются (MIP), т.е.
зелёный — настоящее 3D-пересечение, а не совпадение после проекции.

Выход: out/viz/diagnose/<scan>_<vessel>_clscat.png (3 проекции).

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

import common
from common import (PROJECTION_PANELS, grey_from_hu, view_limits,
                    voxel_in_view)

GT_COLOR = (0.10, 0.35, 0.95)      # синий — GT-only
PRED_COLOR = (0.90, 0.15, 0.15)    # красный — pred-only
OVER_COLOR = (0.10, 0.85, 0.20)    # зелёный — пересечение
CLICK_START = "#FF2D95"
CLICK_END = "#00E5FF"
ALPHA = 0.85


def panel_clscat(grey, gt, pred, affine, drop, row, col):
    """Категориальный MIP: ровно один класс на пиксель (3 цвета)."""
    base, fr, fc = common.orient_mip(grey, drop, row, col, affine)
    ov, _, _ = common.orient_mip((gt & pred).astype(np.float32), drop, row, col, affine)
    po, _, _ = common.orient_mip((pred & ~gt).astype(np.float32), drop, row, col, affine)
    go, _, _ = common.orient_mip((gt & ~pred).astype(np.float32), drop, row, col, affine)
    ov, po, go = ov > 0.5, po > 0.5, go > 0.5
    g = ov
    r = po & ~ov
    b = go & ~ov & ~po
    mask_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
    grey3 = np.stack([base] * 3, axis=-1).astype(np.float32)
    any_ = r | g | b
    rgb = grey3.copy()
    rgb[any_] = (1 - ALPHA) * grey3[any_] + ALPHA * mask_rgb[any_]
    return np.clip(rgb, 0, 1), fr, fc


def render(scan: str, vessel: str, out_dir: Path,
           margin_mm: float = 20.0) -> bool:
    pred_path = common.OUT / "eval" / f"{scan}_{vessel}" / "union.nii.gz"
    if not pred_path.is_file():
        print(f"  нет {pred_path}")
        return False

    ct, affine = common.load_ct(scan)
    gt, _ = common.load_mask(common.fragment_path(scan, vessel))
    pred = np.asanyarray(nib.load(str(pred_path)).dataobj) > 0
    grey = grey_from_hu(ct)
    view = view_limits(gt, affine, margin_mm=margin_mm)
    sample = common.load_sample(scan, vessel)
    clicks = np.array([sample["points"][0]["click_xyz"],
                       sample["points"][1]["click_xyz"]], float)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), dpi=200,
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
        ax = axes[0, ci]
        rgb, fr, fc = panel_clscat(grey, gt, pred, affine, drop, row, col)
        sr = float(abs(affine[row, row]))
        sc = float(abs(affine[col, col]))
        ax.imshow(rgb, origin="lower", interpolation="nearest", aspect=sr / sc)
        ax.set_xlim(*view[col])
        ax.set_ylim(*view[row])
        ax.set_xlabel(f"{clab}, индекс", fontsize=10)
        ax.set_ylabel(f"{rlab}, индекс", fontsize=10)
        ax.tick_params(labelsize=8)
        cc, rr = voxel_in_view(clicks, affine, row, col,
                               rgb.shape[:2], fr, fc)
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
    fig.suptitle(f"{scan} · {vessel.upper()} · clscat: синий / красный / "
                 f"зелёный (1 класс на пиксель)", fontsize=15)
    p = out_dir / f"{scan}_{vessel}_clscat.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  сохранено: {p}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="clscat-визуализация масок")
    ap.add_argument("--scan", action="append", required=True)
    ap.add_argument("--vessel", action="append", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
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
        n += render(scan, vessel, out_dir, args.margin_mm)
    print(f"\nготово: {n} фигур в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
