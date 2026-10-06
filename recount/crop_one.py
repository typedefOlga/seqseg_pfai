"""Обрезка предсказанной маски SeqSeg (union) по двум кликам.

Метод (полная атрибуция вокселей, только предсказание + клики, GT не участвует):
  1) скелетизация маски (skimage.morphology.skeletonize) -> полный скелет;
  2) полный граф скелета: 26-связность, веса рёбер в мм;
  3) полная атрибуция: каждый воксель маски -> ближайший узел полного скелета;
  4) проекция кликов (RAS) -> ближайшие узлы a, b;
  5) геодезический путь P(a->b) по графу;
  6) обрезка: остаются воксели, атрибутированные узлам из P.

Выход:
  recount/out/<scan>_<vessel>_cropped.nii.gz
  recount/out/<scan>_<vessel>_crop.json
  recount/viz/<scan>_<vessel>_crop.png

Запуск:
  python recount/crop_one.py --scan 908 --vessel lcx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

RECOUNT = Path(__file__).resolve().parent
sys.path.insert(0, str(RECOUNT.parent))

import common  # noqa: E402
from common import (PROJECTION_PANELS, grey_from_hu, view_limits,  # noqa: E402
                    voxel_in_view)

GT_COLOR = (0.10, 0.35, 0.95)
PRED_COLOR = (0.90, 0.15, 0.15)
OVER_COLOR = (0.10, 0.85, 0.20)
SKEL_COLOR = "#7A7A7A"
PATH_COLOR = "#FFD400"
CLICK_START = "#FF2D95"
CLICK_END = "#00E5FF"
ALPHA = 0.85


def panel_clscat(grey, gt, pred, affine, drop, row, col):
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


def skeleton_graph(sv, shape, spacing_zyx):
    """Граф 26-связности по вокселям скелета. Возвращает (graph, rows, cols)."""
    n = len(sv)
    _, h, w = shape
    s1, s2 = h * w, w
    keys = sv[:, 0].astype(np.int64) * s1 + sv[:, 1].astype(np.int64) * s2 + sv[:, 2]
    order = np.argsort(keys)
    ks = keys[order]
    offs = [(dz, dy, dx)
            for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dz, dy, dx) > (0, 0, 0)]
    rows, cols, wts = [], [], []
    for dz, dy, dx in offs:
        nk = ((sv[:, 0] + dz).astype(np.int64) * s1
              + (sv[:, 1] + dy).astype(np.int64) * s2 + (sv[:, 2] + dx))
        pos = np.clip(np.searchsorted(ks, nk), 0, n - 1)
        hit = np.nonzero(ks[pos] == nk)[0]
        if hit.size == 0:
            continue
        j = order[pos[hit]]
        rows.append(hit)
        cols.append(j)
        wts.append(np.full(hit.size,
                           float(np.linalg.norm(np.array([dz, dy, dx]) * spacing_zyx))))
    if not rows:
        return coo_matrix((n, n)).tocsr(), np.zeros(0, int), np.zeros(0, int)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    wts = np.concatenate(wts)
    g = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
    return g, rows, cols


def geodesic_path(graph, a, b):
    _, pred = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
    path, cur = [], b
    while cur != -9999 and cur != a:
        path.append(cur)
        cur = int(pred[cur])
    path.append(a)
    return np.array(path[::-1])


def crop_case(scan, vessel):
    affine = nib.load(str(common.ct_path(scan))).affine
    spacing_zyx = tuple(float(z) for z in
                        nib.load(str(common.ct_path(scan))).header.get_zooms()[:3][::-1])
    sample = common.load_sample(scan, vessel)
    clicks_ras = np.array([sample["points"][0]["click_xyz"],
                           sample["points"][1]["click_xyz"]], float)

    gt_zyx = np.transpose(common.load_mask(common.fragment_path(scan, vessel))[0],
                          (2, 1, 0))
    ct, _ = common.load_ct(scan)
    union_xyz = np.asanyarray(nib.load(
        str(common.OUT / "eval" / f"{scan}_{vessel}" / "union.nii.gz")).dataobj) > 0
    union_zyx = np.transpose(union_xyz, (2, 1, 0))

    skel = skeletonize(union_zyx)
    sv = np.argwhere(skel)
    n = len(sv)
    sv_world = common.to_world(affine, sv[:, ::-1])
    tree = cKDTree(sv_world)

    graph, erows, ecols = skeleton_graph(sv, union_zyx.shape, spacing_zyx)

    # проекция кликов на полный скелет
    proj_dist, proj_idx = tree.query(clicks_ras)
    a, b = int(proj_idx[0]), int(proj_idx[1])
    path = geodesic_path(graph, a, b)
    pathset = np.zeros(n, dtype=bool)
    pathset[path] = True

    # полная атрибуция вокселей маски ближайшему узлу скелета
    mv = np.argwhere(union_zyx)
    mv_world = common.to_world(affine, mv[:, ::-1])
    _, mi = tree.query(mv_world)
    keep = pathset[mi]

    cropped_zyx = np.zeros_like(union_zyx)
    cropped_zyx[tuple(mv[keep].T)] = True
    cropped_xyz = np.transpose(cropped_zyx, (2, 1, 0))

    # честность: попадают ли клики в предсказанную маску
    def click_in_mask(world):
        v = np.rint(common.to_voxel(affine, world)).astype(int)  # x,y,z
        if any(v[i] < 0 or v[i] >= union_xyz.shape[i] for i in range(3)):
            return False
        return bool(union_xyz[v[0], v[1], v[2]])

    path_world = sv_world[path] if len(path) else np.zeros((0, 3))
    path_arc = float(np.linalg.norm(np.diff(path_world, axis=0), axis=1).sum()) \
        if len(path_world) > 1 else 0.0
    start_in = click_in_mask(clicks_ras[0])
    end_in = click_in_mask(clicks_ras[1])
    warnings = []
    if not start_in:
        warnings.append(f"start click вне маски (proj {proj_dist[0]:.1f} мм)")
    if not end_in:
        warnings.append(f"end click вне маски (proj {proj_dist[1]:.1f} мм)")
    if path_arc < 1.0:
        warnings.append(f"вырожденный путь клик-клик ({path_arc:.2f} мм)")

    info = {
        "scan": scan, "vessel": vessel,
        "n_skel_nodes": int(n), "n_path_nodes": int(len(path)),
        "path_arc_mm": round(path_arc, 2),
        "proj_dist_mm": [round(float(proj_dist[0]), 2), round(float(proj_dist[1]), 2)],
        "start_click_in_mask": bool(start_in),
        "end_click_in_mask": bool(end_in),
        "union_voxels": int(union_xyz.sum()),
        "cropped_voxels": int(cropped_xyz.sum()),
        "gt_voxels": int(gt_zyx.sum()),
        "removed_frac": round(float(1.0 - cropped_xyz.sum() / max(union_xyz.sum(), 1)), 4),
        "warnings": warnings,
    }
    return {
        "union_xyz": union_xyz, "cropped_xyz": cropped_xyz, "gt_xyz": np.transpose(
            gt_zyx, (2, 1, 0)),
        "ct": ct, "affine": affine,
        "sv_world": sv_world, "edges": (erows, ecols),
        "path_world": path_world, "clicks_ras": clicks_ras,
        "proj_world": sv_world[proj_idx], "info": info,
    }


def render(scan, vessel, d, out_png, margin_mm=20.0):
    grey = grey_from_hu(d["ct"])
    union_xyz, cropped_xyz, gt_xyz = d["union_xyz"], d["cropped_xyz"], d["gt_xyz"]
    affine = d["affine"]
    view = view_limits(union_xyz | gt_xyz | cropped_xyz, affine, margin_mm=margin_mm)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.2), dpi=180,
                             layout="constrained", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.0, hspace=0.02)
    rows = (("исходная union vs GT", union_xyz),
            ("обрезанная по кликам vs GT", cropped_xyz))
    sw, (er, ec) = d["sv_world"], d["edges"]

    for ri, (rtitle, pred) in enumerate(rows):
        for ci, (title, drop, row, col, rlab, clab) in enumerate(PROJECTION_PANELS):
            ax = axes[ri, ci]
            rgb, fr, fc = panel_clscat(grey, gt_xyz, pred, affine, drop, row, col)
            sr = float(abs(affine[row, row]))
            sc = float(abs(affine[col, col]))
            ax.imshow(rgb, origin="lower", interpolation="nearest", aspect=sr / sc)
            ax.set_xlim(*view[col])
            ax.set_ylim(*view[row])
            ax.tick_params(labelsize=7)
            ax.set_xlabel(f"{clab}, индекс", fontsize=9)

            # весь граф-скелет (все ветви)
            if len(er):
                ew = sw[er]
                ewc, ewr = voxel_in_view(np.vstack([sw[er], sw[ec]]), affine,
                                         row, col, rgb.shape[:2], fr, fc)
                half = len(er)
                segs = np.stack([np.stack([ewc[:half], ewr[:half]], axis=1),
                                 np.stack([ewc[half:], ewr[half:]], axis=1)],
                                axis=1)
                ax.add_collection(LineCollection(segs, colors=SKEL_COLOR,
                                                 linewidths=0.6, alpha=0.85,
                                                 zorder=5))
            # оставленный путь клик-клик
            if len(d["path_world"]) > 1:
                pc, pr = voxel_in_view(d["path_world"], affine, row, col,
                                       rgb.shape[:2], fr, fc)
                ax.plot(pc, pr, "-", color=PATH_COLOR, lw=1.6, alpha=0.98, zorder=6)

            pc, pr = voxel_in_view(d["proj_world"], affine, row, col,
                                   rgb.shape[:2], fr, fc)
            ax.plot(pc[0], pr[0], "X", color=CLICK_START, ms=9, mec="black",
                    mew=0.8, zorder=8)
            ax.plot(pc[1], pr[1], "X", color=CLICK_END, ms=9, mec="black",
                    mew=0.8, zorder=8)
            kc, kr = voxel_in_view(d["clicks_ras"], affine, row, col,
                                   rgb.shape[:2], fr, fc)
            ax.plot(kc[0], kr[0], "x", color=CLICK_START, ms=6, mew=1.2, zorder=8)
            ax.plot(kc[1], kr[1], "x", color=CLICK_END, ms=6, mew=1.2, zorder=8)

            ax.set_title(f"{rtitle}: {title}", fontsize=11)
            if ri == 1:
                ax.set_ylabel(f"{rlab}, индекс", fontsize=9)

    handles = [Patch(facecolor=GT_COLOR, edgecolor="black", label="GT-only (синий)"),
               Patch(facecolor=PRED_COLOR, edgecolor="black", label="pred-only (красный)"),
               Patch(facecolor=OVER_COLOR, edgecolor="black", label="пересечение (зелёный)"),
               Line2D([0], [0], color=SKEL_COLOR, lw=1.0, label="граф-скелет (все ветви)"),
               Line2D([0], [0], color=PATH_COLOR, lw=1.8, label="путь клик-клик (оставлен)"),
               Line2D([0], [0], marker="X", color="none", mec="black",
                      mfc=CLICK_START, ls="none", ms=9, label="проекция клика старта"),
               Line2D([0], [0], marker="X", color="none", mec="black",
                      mfc=CLICK_END, ls="none", ms=9, label="проекция клика конца"),
               Line2D([0], [0], marker="x", color=CLICK_START, ls="none", ms=7,
                      label="сырой клик старта"),
               Line2D([0], [0], marker="x", color=CLICK_END, ls="none", ms=7,
                      label="сырой клик конца")]
    info = d["info"]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, fontsize=9)
    warn = (" | WARN: " + "; ".join(info["warnings"])) if info["warnings"] else ""
    fig.suptitle(
        f"{scan} · {vessel.upper()} · полная атрибуция: union "
        f"{info['union_voxels']} -> {info['cropped_voxels']} вокс. "
        f"(GT {info['gt_voxels']}); путь {info['path_arc_mm']:.1f} мм{warn}",
        fontsize=14)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Обрезка union-маски по двум кликам")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--vessel", required=True)
    ap.add_argument("--outdir", default=str(RECOUNT / "out"))
    ap.add_argument("--vizdir", default=str(RECOUNT / "viz"))
    ap.add_argument("--margin-mm", type=float, default=20.0)
    args = ap.parse_args()
    scan, vessel = args.scan, args.vessel

    d = crop_case(scan, vessel)
    info = d["info"]

    out_dir = Path(args.outdir)
    viz_dir = Path(args.vizdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    nii = out_dir / f"{scan}_{vessel}_cropped.nii.gz"
    nib.save(nib.Nifti1Image(d["cropped_xyz"].astype(np.uint8), d["affine"]),
             str(nii))
    js = out_dir / f"{scan}_{vessel}_crop.json"
    js.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    png = viz_dir / f"{scan}_{vessel}_crop.png"
    render(scan, vessel, d, png, args.margin_mm)

    print(f"скан {scan} · {vessel.upper()}")
    print(f"  узлов скелета / на пути : {info['n_skel_nodes']} / "
          f"{info['n_path_nodes']} (путь {info['path_arc_mm']:.1f} мм)")
    print(f"  расстояния проекций     : {info['proj_dist_mm'][0]} , "
          f"{info['proj_dist_mm'][1]} мм")
    print(f"  клики в маске           : start={info['start_click_in_mask']} "
          f"end={info['end_click_in_mask']}")
    print(f"  воксели                 : union {info['union_voxels']} -> "
          f"cropped {info['cropped_voxels']} "
          f"(срезано {info['removed_frac']*100:.1f}%); GT {info['gt_voxels']}")
    if info["warnings"]:
        print("  WARN: " + "; ".join(info["warnings"]))
    print(f"  маска     : {nii}\n  json      : {js}\n  визуализация: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
