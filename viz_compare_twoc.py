"""Сравнение двух схем раскраски масок (строки = схемы, столбцы = 3 проекции).

По умолчанию сравниваются clscat (категориально, 3 цвета) и twoc (R/B).
Отдельные файлы, ничего не перезаписывает.

Два режима:
  1) по явным кейсам:
       python viz_compare_twoc.py --scan 129 --vessel lad
     -> out/viz/diagnose/<scan>_<vessel>_compare_<a>_<b>.png
  2) по перцентилям Dice (p5/p50/p95), как `viz_percentiles.py`:
       python viz_compare_twoc.py --percentiles
     -> out/viz/statistic/<vessel>_<pXX>_compare_<a>_<b>.png

Пояснение к схеме:
  twoc   — маски проецируются раздельно, затем складываются в 2D:
           маджента = (exists z pred) & (exists z gt) — совпадение по лучу.
  clscat — сначала пересечение по вокселям, затем проекция:
           зелёный = exists z (pred & gt) — настоящее 3D-пересечение.
  Поэтому зелёный ⊆ маджента: лишняя маджента = разные глубины в одном луче.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import common
from common import (PROJECTION_PANELS, grey_from_hu, view_limits, voxel_in_view)
import viz_diagnose as V
import viz_percentiles as P

TAGMAP = {
    "clscat": (V.panel_clscat,
               "clscat: синий / красный / зелёный (1 класс на пиксель)"),
    "twoc": (V.panel_twoc, "twoc (R/B): синий / красный / маджента"),
    "twocA": (V.panel_classes, "twocA (R/G/B): синий / красный / зелёный"),
    "channels": (V.panel_channels,
                 "channels: GT синий / pred жёлтый / пересечение зелёный"),
    "slab": (V.panel_tpfn, "slab: TP / FN / FP"),
    "outline": (V.panel_outline, "outline: контуры GT / pred"),
}


def render_compare(scan: str, vessel: str, out_path: Path, tag_a: str,
                   tag_b: str, margin_mm: float = 20.0,
                   title_extra: str = "") -> bool:
    """Одна фигура: 2 строки (схемы) x 3 проекции. True, если сохранено."""
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

    rows = [TAGMAP[tag_a], TAGMAP[tag_b]]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.2), dpi=200,
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.0)
    for ri, (fn, label) in enumerate(rows):
        for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
            ax = axes[ri, ci]
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
            cc, rr = voxel_in_view(clicks, affine, row, col,
                                   rgb.shape[:2], fr, fc)
            ax.plot(cc[0], rr[0], "X", color=V.CLICK_START, ms=10,
                    mec="black", mew=1.0, zorder=7)
            ax.plot(cc[1], rr[1], "X", color=V.CLICK_END, ms=10,
                    mec="black", mew=1.0, zorder=7)
            if ri == 0:
                ax.set_title(title, fontsize=13)
        axes[ri, 0].text(0.02, 0.98, label, transform=axes[ri, 0].transAxes,
                         va="top", ha="left", fontsize=11, weight="bold",
                         bbox=dict(facecolor="white", alpha=0.7, pad=2))

    fig.suptitle(f"{scan} · {vessel.upper()}{title_extra} · "
                 f"{tag_a} vs {tag_b}", fontsize=15)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  сохранено: {out_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение двух схем раскраски")
    ap.add_argument("--scan", action="append", default=None)
    ap.add_argument("--vessel", action="append", default=None)
    ap.add_argument("--percentiles", action="store_true",
                    help="кейсы p5/p50/p95 по каждой артерии (как "
                         "viz_percentiles.py), а не явные --scan/--vessel")
    ap.add_argument("--metrics", default=None,
                    help="CSV с метриками для --percentiles")
    ap.add_argument("--tag-a", default="clscat", choices=list(TAGMAP))
    ap.add_argument("--tag-b", default="twoc", choices=list(TAGMAP))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    args = ap.parse_args()

    if args.percentiles:
        metrics_path = (Path(args.metrics) if args.metrics
                        else common.OUT / "reports" / "metrics_union.csv")
        out_dir = (Path(args.outdir) if args.outdir
                   else common.OUT / "viz" / "statistic")
        out_dir.mkdir(parents=True, exist_ok=True)
        selected = P.select_cases(metrics_path)
        n = 0
        for vessel in P.VESSELS:
            if vessel not in selected:
                continue
            for q, label in P.PERCENTILES:
                case, dice, pv = selected[vessel][label]
                scan = case.rsplit("_", 1)[0]
                extra = f" · {label} (p{q}={pv:.3f}) · dice={dice:.3f}"
                out_path = (out_dir / f"{vessel}_{label}_compare_"
                                      f"{args.tag_a}_{args.tag_b}.png")
                n += render_compare(scan, vessel, out_path, args.tag_a,
                                    args.tag_b, args.margin_mm, extra)
        print(f"\nготово: {n} фигур в {out_dir}")
        return 0

    if not args.scan or not args.vessel:
        print("нужны --scan/--vessel или --percentiles")
        return 1
    if len(args.scan) != len(args.vessel):
        print("число --scan и --vessel должно совпадать")
        return 1
    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "viz" / "diagnose")
    out_dir.mkdir(parents=True, exist_ok=True)
    for scan, vessel in zip(args.scan, args.vessel):
        out_path = (out_dir / f"{scan}_{vessel}_compare_"
                              f"{args.tag_a}_{args.tag_b}.png")
        render_compare(scan, vessel, out_path, args.tag_a, args.tag_b,
                       args.margin_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
