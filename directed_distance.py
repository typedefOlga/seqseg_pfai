"""Направленное расстояние от GT-маски до предсказания (GT -> pred).

Для каждого уже посчитанного кейса (union.nii.gz) считается расстояние (мм) от
поверхности GT до ближайшего вокселя предсказания; берётся супремум
(направленный HD GT->pred) и его 95-й процентиль.

Считается в bbox(pred ∪ gt) — как метрики в evaluate.py. Формула совпадает с
одним из направлений hd95: d_gp = EDT(~pred)[поверхность GT].

Выход (в отдельные файлы, metrics_union.csv/metrics_mean.csv не трогаются):
  out/reports/metrics_gt_to_pred.csv       — те же колонки, что в
        metrics_union.csv, + gt_to_pred_max, gt_to_pred_p95 (per-case).
  out/reports/metrics_gt_to_pred_mean.csv  — тот же формат, что в
        metrics_mean.csv (group,n,metric,mean,std). Для каждой группы и обеих
        метрик: mean/std по снимкам; а также агрегаты по снимкам — супремум
        (gt_to_pred_max@sup) и 95-й процентиль (gt_to_pred_max@p95).

Запуск:
    python directed_distance.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import common
from evaluate import crop_to_union

S6 = ndimage.generate_binary_structure(3, 1)
NEW_COLS = ("gt_to_pred_max", "gt_to_pred_p95")
GROUPS = ("all", "LAD", "LCX", "RCA")


def directed_gt_to_pred(scan: str, vessel: str):
    """(sup, p95) расстояний от поверхности GT до ближайшего вокселя pred (мм)."""
    pred_path = common.OUT / "eval" / f"{scan}_{vessel}" / "union.nii.gz"
    pimg = nib.load(str(pred_path))
    pred = np.transpose(np.asanyarray(pimg.dataobj) > 0, (2, 1, 0))  # (z,y,x)
    gt_xyz, _ = common.load_mask(common.fragment_path(scan, vessel))
    gt = np.transpose(gt_xyz, (2, 1, 0))
    spacing_zyx = tuple(float(z) for z in pimg.header.get_zooms()[:3][::-1])

    pc, gc, _ = crop_to_union(pred, gt, pimg.affine)
    if not pc.any() or not gc.any():
        return float("nan"), float("nan")

    dt_pred = ndimage.distance_transform_edt(~pc, sampling=spacing_zyx)
    surf_g = gc & ~ndimage.binary_erosion(gc, structure=S6, border_value=0)
    d = dt_pred[surf_g]
    if d.size == 0:
        return float("nan"), float("nan")
    return float(d.max()), float(np.percentile(d, 95))


def write_per_case(metrics_path: Path, out_path: Path) -> dict:
    """Дописывает две колонки к metrics_union.csv; возвращает значения done-строк."""
    with open(metrics_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_cols = list(reader.fieldnames)
        rows = list(reader)

    values = {}
    n = 0
    for r in rows:
        if r.get("status") != "done":
            r.update({c: "" for c in NEW_COLS})
            continue
        key = (r["scan"], r["vessel"])
        mx, p95 = directed_gt_to_pred(r["scan"], r["vessel"])
        values[(r["scan"], r["vessel"].upper())] = (mx, p95)
        r["gt_to_pred_max"] = round(mx, 6)
        r["gt_to_pred_p95"] = round(p95, 6)
        n += 1
        if n % 20 == 0:
            print(f"  ... {n} кейсов")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_cols + list(NEW_COLS))
        w.writeheader()
        w.writerows(rows)
    print(f"  сохранено: {out_path} ({n} кейсов)")
    return values


def write_mean(values: dict, out_path: Path) -> None:
    """Сводка в формате metrics_mean.csv (group,n,metric,mean,std)."""
    rows = []
    for g in GROUPS:
        if g == "all":
            items = list(values.values())
        else:
            items = [v for (_, vessel), v in values.items() if vessel == g]
        n = len(items)
        if n == 0:
            continue
        arr_max = np.array([v[0] for v in items], float)
        arr_p95 = np.array([v[1] for v in items], float)
        rows.append((g, n, "gt_to_pred_max", arr_max.mean(), arr_max.std()))
        rows.append((g, n, "gt_to_pred_p95", arr_p95.mean(), arr_p95.std()))
        # агрегаты по снимкам (значение в колонке mean, std пустой)
        rows.append((g, n, "gt_to_pred_max@sup", arr_max.max(), ""))
        rows.append((g, n, "gt_to_pred_max@p95",
                     float(np.percentile(arr_max, 95)), ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "n", "metric", "mean", "std"])
        for g, n, metric, mean, std in rows:
            mean = round(float(mean), 6)
            std = "" if std == "" else round(float(std), 6)
            w.writerow([g, n, metric, mean, std])
    print(f"  сохранено: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="HD GT->pred из готовых масок")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-mean", default=None)
    args = ap.parse_args()

    metrics_path = (Path(args.metrics) if args.metrics
                    else common.OUT / "reports" / "metrics_union.csv")
    out_path = (Path(args.out) if args.out
                else common.OUT / "reports" / "metrics_gt_to_pred.csv")
    out_mean = (Path(args.out_mean) if args.out_mean
                else common.OUT / "reports" / "metrics_gt_to_pred_mean.csv")

    print(f"читаю {metrics_path}")
    values = write_per_case(metrics_path, out_path)
    write_mean(values, out_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
