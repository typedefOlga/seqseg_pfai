"""Метрики сегментации для задачи «2 точки -> фрагмент сосуда».

Маска: dice, precision/recall/f1, cl_dice, hd, hd95, assd, chamfer,
surface_dice@tol, boundary_f1/precision/recall@tol.
Топология (b012): b0, b1, b2 (числа Бетти) и их ошибка относительно GT.
Центрлиния: coverage, centerline_md, centerline_hd95.
Радиус: radius_mae/rmse/bias (EDT предсказания в точках GT-центрлинии).

Все расстояния — в мм, с учётом spacing. Маски — bool-массивы (z, y, x).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

S6 = ndimage.generate_binary_structure(3, 1)
S26 = ndimage.generate_binary_structure(3, 3)


# --------------------------------------------------------------------------- #
#  Маска: пересечение / базовые                                            #
# --------------------------------------------------------------------------- #

def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = pred.astype(bool), gt.astype(bool)
    denom = int(pred.sum()) + int(gt.sum())
    return 1.0 if denom == 0 else float(2 * (pred & gt).sum() / denom)


def prf(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(gt.sum()), 1)
    f = 2 * p * r / max(p + r, 1e-12)
    return {"precision": p, "recall": r, "f1": f}


def cl_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Centreline Dice (Shit et al. 2021) — как в бенчмарке ImageCAS-X."""
    from skimage.morphology import skeletonize

    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return 0.0
    sp, sg = skeletonize(pred), skeletonize(gt)
    t_prec = (sp & gt).sum() / max(sp.sum(), 1)
    t_sen = (sg & pred).sum() / max(sg.sum(), 1)
    return 0.0 if (t_prec + t_sen) == 0 else float(
        2 * t_prec * t_sen / (t_prec + t_sen))


# --------------------------------------------------------------------------- #
#  Поверхности                                                               #
# --------------------------------------------------------------------------- #

def surface(mask: np.ndarray) -> np.ndarray:
    """Граница: маска минус её 6-связная эрозия."""
    return mask.astype(bool) & ~ndimage.binary_erosion(
        mask.astype(bool), structure=S6, border_value=0)


def gt_cache(gt: np.ndarray, spacing_zyx) -> dict:
    """Предрасчёт GT-части для surface_metrics: EDT и поверхность (один раз на кейс)."""
    gt = gt.astype(bool)
    return {
        "sp": tuple(float(s) for s in spacing_zyx),
        "dt_gt": ndimage.distance_transform_edt(~gt, sampling=tuple(
            float(s) for s in spacing_zyx)),
        "surf_g": surface(gt),
    }


def surface_metrics(pred: np.ndarray, gt: np.ndarray, spacing_zyx,
                    tols=(0.12, 0.5, 1.0), cache: dict | None = None) -> dict:
    """HD, HD95, ASSD, chamfer, surface Dice и boundary F1 при допусках tols."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    out = {}
    if not pred.any() or not gt.any():
        for k in ("hd", "hd95", "assd", "chamfer", "surface_dice"):
            out[k] = float("nan")
        for t in tols:
            out[f"boundary_f1@{t}"] = float("nan")
            out[f"boundary_precision@{t}"] = float("nan")
            out[f"boundary_recall@{t}"] = float("nan")
        return out

    if cache is not None:
        sp, dt_gt, surf_g = cache["sp"], cache["dt_gt"], cache["surf_g"]
    else:
        sp = tuple(float(s) for s in spacing_zyx)
        dt_gt = ndimage.distance_transform_edt(~gt, sampling=sp)
        surf_g = surface(gt)
    dt_pred = ndimage.distance_transform_edt(~pred, sampling=sp)
    surf_p = surface(pred)
    d_pg = dt_gt[surf_p]      # расстояния с поверхности pred до GT
    d_gp = dt_pred[surf_g]    # расстояния с поверхности GT до pred

    out["hd"] = float(max(d_pg.max(), d_gp.max()))
    out["hd95"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
    out["assd"] = float((d_pg.mean() + d_gp.mean()) / 2.0)

    # chamfer через KD-дерево физических координат (симметричное среднее)
    pts_p = np.argwhere(surf_p) * np.asarray(sp)
    pts_g = np.argwhere(surf_g) * np.asarray(sp)
    ch_pg = cKDTree(pts_g).query(pts_p)[0]
    ch_gp = cKDTree(pts_p).query(pts_g)[0]
    out["chamfer"] = float((ch_pg.mean() + ch_gp.mean()) / 2.0)

    for t in tols:
        bp = float((d_pg <= t).mean()) if d_pg.size else 0.0
        br = float((d_gp <= t).mean()) if d_gp.size else 0.0
        out[f"boundary_precision@{t}"] = bp
        out[f"boundary_recall@{t}"] = br
        out[f"boundary_f1@{t}"] = (2 * bp * br / (bp + br)
                                   if (bp + br) > 0 else 0.0)
        sd = ((d_pg <= t).sum() + (d_gp <= t).sum()) / max(
            d_pg.size + d_gp.size, 1)
        out.setdefault("surface_dice", {})[t] = float(sd)
    return out


# --------------------------------------------------------------------------- #
#  Топология: числа Бетти b0, b1, b2                                          #
# --------------------------------------------------------------------------- #

def betti_numbers(mask: np.ndarray) -> tuple[int, int, int]:
    """(b0, b1, b2): компоненты (26), петли, замкнутые полости (6-связный фон)."""
    from skimage.measure import euler_number

    mask = mask.astype(bool)
    if not mask.any():
        return 0, 0, 0
    _, b0 = ndimage.label(mask, structure=S26)

    bg = ~mask
    lab_b, n_b = ndimage.label(bg, structure=S6)
    border = set(lab_b[0].ravel()) | set(lab_b[-1].ravel()) | \
        set(lab_b[:, 0].ravel()) | set(lab_b[:, -1].ravel()) | \
        set(lab_b[:, :, 0].ravel()) | set(lab_b[:, :, -1].ravel())
    border.discard(0)
    b2 = int(n_b - len(border))

    euler = int(euler_number(mask, connectivity=3))
    b1 = int(b0 + b2 - euler)
    return int(b0), b1, b2


def betti_metrics(pred: np.ndarray, gt: np.ndarray,
                  gt_b: tuple | None = None) -> dict:
    bp = betti_numbers(pred)
    bg = tuple(gt_b) if gt_b is not None else betti_numbers(gt)
    return {
        "b0_pred": bp[0], "b1_pred": bp[1], "b2_pred": bp[2],
        "b0_gt": bg[0], "b1_gt": bg[1], "b2_gt": bg[2],
        "b0_err": abs(bp[0] - bg[0]),
        "b1_err": abs(bp[1] - bg[1]),
        "b2_err": abs(bp[2] - bg[2]),
    }


# --------------------------------------------------------------------------- #
#  Центрлиния                                                                #
# --------------------------------------------------------------------------- #

def centerline_from_mask(mask: np.ndarray, spacing_zyx) -> np.ndarray:
    """Упорядоченная центрлиния маски (скелет + самый длинный путь), (z,y,x).

    Рёбра скелета строятся векторно по 13 каноническим 26-соседям (через
    searchsorted по ключам вокселей) — быстро даже на больших масках.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from skimage.morphology import skeletonize

    mask = mask.astype(bool)
    skel = skeletonize(mask)
    vox = np.argwhere(skel)
    n = len(vox)
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return vox.astype(float)
    sp = np.asarray(spacing_zyx, dtype=float)

    d, h, w = mask.shape
    s1, s2 = h * w, w
    keys = vox[:, 0].astype(np.int64) * s1 + vox[:, 1].astype(np.int64) * s2 \
        + vox[:, 2]
    order = np.argsort(keys)
    keys_sorted = keys[order]

    offsets = [(dz, dy, dx)
               for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dz, dy, dx) > (0, 0, 0)]
    rows, cols, wts = [], [], []
    for dz, dy, dx in offsets:
        nk = (vox[:, 0] + dz).astype(np.int64) * s1 \
            + (vox[:, 1] + dy).astype(np.int64) * s2 + (vox[:, 2] + dx)
        pos = np.clip(np.searchsorted(keys_sorted, nk), 0, n - 1)
        hit = np.nonzero(keys_sorted[pos] == nk)[0]
        if hit.size == 0:
            continue
        j = order[pos[hit]]
        rows.append(hit)
        cols.append(j)
        wts.append(np.full(hit.size, float(np.linalg.norm(
            np.array([dz, dy, dx]) * sp))))
    if not rows:
        return vox.astype(float)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    wts = np.concatenate(wts)
    g = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()
    d0 = dijkstra(g, directed=False, indices=0)
    a = int(np.argmax(np.where(np.isfinite(d0), d0, -1.0)))
    d_a, pred_a = dijkstra(g, directed=False, indices=a,
                           return_predecessors=True)
    b = int(np.argmax(np.where(np.isfinite(d_a), d_a, -1.0)))
    path, cur = [], b
    while cur != -9999 and cur != a:
        path.append(cur)
        cur = int(pred_a[cur])
    path.append(a)
    return vox[np.array(path[::-1])].astype(float)


def _symmetric_pt_distances(a, b) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")
    d_ab = cKDTree(b).query(a)[0]
    d_ba = cKDTree(a).query(b)[0]
    md = float((d_ab.mean() + d_ba.mean()) / 2.0)
    hd95 = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    return md, hd95


def centerline_metrics(pred_cl_world: np.ndarray,
                       gt_cl_world: np.ndarray,
                       tol_mm: float = 1.0) -> dict:
    """Симметричные среднее и hd95 между центролиниями (мм) + покрытие GT."""
    md, hd95 = _symmetric_pt_distances(pred_cl_world, gt_cl_world)
    if len(gt_cl_world) == 0:
        cov = float("nan")
    elif len(pred_cl_world) == 0:
        cov = 0.0
    else:
        d = cKDTree(pred_cl_world).query(gt_cl_world)[0]
        cov = float((d <= tol_mm).mean())
    return {"centerline_md": md, "centerline_hd95": hd95,
            f"coverage@{tol_mm}mm": cov}


# --------------------------------------------------------------------------- #
#  Радиус                                                                    #
# --------------------------------------------------------------------------- #

def radius_metrics(pred: np.ndarray, spacing_zyx,
                   gt_cl_world: np.ndarray, gt_radius: np.ndarray,
                   affine, max_near_mm: float = 3.0) -> dict:
    """Сравнение радиуса: EDT предсказания в точках GT-центрлинии vs GT-радиус."""
    if not pred.any() or len(gt_cl_world) == 0:
        return {"radius_mae": float("nan"), "radius_rmse": float("nan"),
                "radius_bias": float("nan")}
    sp = np.asarray(spacing_zyx, dtype=float)
    edt = ndimage.distance_transform_edt(pred, sampling=tuple(sp))

    from common import to_voxel
    vox = to_voxel(affine, gt_cl_world)          # (x,y,z)
    xyz = np.rint(vox).astype(int)
    ok = np.ones(len(xyz), dtype=bool)
    for i in range(3):
        ok &= (xyz[:, i] >= 0) & (xyz[:, i] < pred.shape[2 - i])
    if not ok.any():
        return {"radius_mae": float("nan"), "radius_rmse": float("nan"),
                "radius_bias": float("nan")}
    zz, yy, xx = xyz[ok, 2], xyz[ok, 1], xyz[ok, 0]
    pred_r = edt[zz, yy, xx]
    gt_r = np.asarray(gt_radius)[ok]
    near = pred_r <= max_near_mm
    if not near.any():
        return {"radius_mae": float("nan"), "radius_rmse": float("nan"),
                "radius_bias": float("nan")}
    # радиус сосуда в точке трассы = ближайшая ненулевая дистанция до границы
    d = pred_r[near].copy()
    d[d <= 0] = np.nan
    diff = d - gt_r[near]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return {"radius_mae": float("nan"), "radius_rmse": float("nan"),
                "radius_bias": float("nan")}
    return {"radius_mae": float(np.abs(diff).mean()),
            "radius_rmse": float(np.sqrt((diff ** 2).mean())),
            "radius_bias": float(diff.mean())}


# --------------------------------------------------------------------------- #
#  Сводная оценка                                                            #
# --------------------------------------------------------------------------- #

def evaluate_mask(pred: np.ndarray, gt: np.ndarray, spacing_zyx,
                  gt_cl_world: np.ndarray | None = None,
                  gt_radius: np.ndarray | None = None,
                  affine=None, scache: dict | None = None,
                  gt_b: tuple | None = None) -> dict:
    """Полный набор метрик pred vs gt (bool (z,y,x)) + центрлиния/радиус."""
    res = {"pred_voxels": int(pred.sum()), "gt_voxels": int(gt.sum())}
    res["dice"] = dice(pred, gt)
    res.update(prf(pred, gt))
    res["cl_dice"] = cl_dice(pred, gt)
    res.update(surface_metrics(pred, gt, spacing_zyx, cache=scache))
    res.update(betti_metrics(pred, gt, gt_b=gt_b))

    if gt_cl_world is not None:
        cl_vox = centerline_from_mask(pred, spacing_zyx)
        if len(cl_vox) and affine is not None:
            from common import to_world
            pred_cl_world = to_world(affine, cl_vox[:, ::-1])
        else:
            pred_cl_world = np.zeros((0, 3))
        res.update(centerline_metrics(pred_cl_world, gt_cl_world))
        if gt_radius is not None and affine is not None:
            res.update(radius_metrics(pred, spacing_zyx, gt_cl_world,
                                      gt_radius, affine))
    return res
