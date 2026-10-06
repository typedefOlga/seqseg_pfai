"""Относительная ошибка радиуса предсказания (click-predict маски) vs GT.

Для каждого кейса берётся финализированная маска предсказания
(out/eval/<scan>_<vessel>/union.nii.gz = click-crop), считается EDT
предсказания, радиус `r_pred` берётся в точках GT-центрлинии (из sample),
`r_gt` — `centerline[].radius_mm` из sample. Относительная ошибка:

    e = (r_pred - r_gt) / r_gt

Метрики на множестве точек, отфильтрованном как в metrics.radius_metrics
(0 < r_pred <= max_near_mm, r_gt >= min_r_gt):
    rel_mse  = mean(e^2)
    rel_rmse = sqrt(mean(e^2))
    rel_bias = mean(e)
    rel_mae  = mean(|e|)
    rel_p95  = 95-й перцентиль |e|

Выход:
    out/reports/metrics_radius_relative.csv       — по строке на кейс
    out/reports/metrics_radius_relative_mean.csv  — сводка (group,n,metric,mean,std)

Запуск:
    SEQSEG_OUT=out0610 python radius_relative.py
    python radius_relative.py --cases cases_120_50.txt
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

import common
from common import to_voxel
from evaluate import crop_to_union

CASE_COLUMNS = [
    "scan", "vessel", "status", "n_points",
    "rel_mse", "rel_rmse", "rel_bias", "rel_mae", "rel_p95",
]
GROUP_KEY = ["rel_mse", "rel_rmse", "rel_bias", "rel_mae", "rel_p95"]


def read_cases(path: Path) -> list[tuple[str, str]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scan, vessel = line.split()
        cases.append((scan, vessel.lower()))
    return cases


def relative_radius(scan: str, vessel: str, max_near_mm: float,
                    min_r_gt: float) -> dict:
    """Относительные метрики радиуса для одного кейса."""
    pred_path = common.OUT / "eval" / f"{scan}_{vessel}" / "union.nii.gz"
    if not pred_path.is_file():
        return {"status": "no_mask"}

    pimg = nib.load(str(pred_path))
    pred_xyz = np.asanyarray(pimg.dataobj) > 0        # (x, y, z)
    affine = pimg.affine
    spacing_zyx = tuple(float(z) for z in pimg.header.get_zooms()[:3][::-1])
    sample = common.load_sample(scan, vessel)
    gt_cl_world = np.array([c["xyz"] for c in sample["centerline"]], float)
    gt_radius = np.array([c["radius_mm"] for c in sample["centerline"]], float)

    if not pred_xyz.any() or len(gt_cl_world) == 0:
        return {"status": "empty"}

    gt_xyz, _ = common.load_mask(common.fragment_path(scan, vessel))
    pred = np.transpose(pred_xyz, (2, 1, 0))          # (z, y, x)
    gt = np.transpose(gt_xyz, (2, 1, 0))
    pc, _gc, aff = crop_to_union(pred, gt, affine)    # только bbox(pred∪gt)

    edt = ndimage.distance_transform_edt(pc, sampling=spacing_zyx)
    vox = to_voxel(aff, gt_cl_world)                  # (x, y, z)
    xyz = np.rint(vox).astype(int)
    ok = np.ones(len(xyz), dtype=bool)
    ok &= (xyz[:, 0] >= 0) & (xyz[:, 0] < pc.shape[2])
    ok &= (xyz[:, 1] >= 0) & (xyz[:, 1] < pc.shape[1])
    ok &= (xyz[:, 2] >= 0) & (xyz[:, 2] < pc.shape[0])
    if not ok.any():
        return {"status": "no_points"}

    r_pred = edt[xyz[ok, 2], xyz[ok, 1], xyz[ok, 0]]
    r_gt = gt_radius[ok]
    near = (r_pred > 0) & (r_pred <= max_near_mm) & (r_gt >= min_r_gt)
    if not near.any():
        return {"status": "no_points"}

    e = (r_pred[near] - r_gt[near]) / r_gt[near]
    return {
        "status": "done",
        "n_points": int(e.size),
        "rel_mse": float(np.mean(e ** 2)),
        "rel_rmse": float(np.sqrt(np.mean(e ** 2))),
        "rel_bias": float(np.mean(e)),
        "rel_mae": float(np.mean(np.abs(e))),
        "rel_p95": float(np.percentile(np.abs(e), 95)),
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups = {"all": rows}
    for r in rows:
        if r.get("status") == "done":
            groups.setdefault(r["vessel"].upper(), []).append(r)
    out = []
    for gname, grp in groups.items():
        if gname != "all":
            grp = [r for r in grp if r.get("status") == "done"]
        n = len(grp)
        if n == 0:
            continue
        for metric in GROUP_KEY:
            vals = [float(r[metric]) for r in grp
                    if r.get(metric) not in (None, "", "None")]
            if not vals:
                continue
            out.append({"group": gname, "n": n, "metric": metric,
                        "mean": round(sum(vals) / len(vals), 5),
                        "std": round(st.pstdev(vals), 5) if len(vals) > 1
                        else 0.0})
    return out


def print_table(per_case: list[dict]) -> None:
    done = [r for r in per_case if r.get("status") == "done"]
    print(f"{'artery':8}{'n':>4}{'rel_mse':>10}{'rel_rmse':>11}"
          f"{'rel_bias':>10}{'rel_mae':>9}{'rel_p95':>9}")
    vessels = ["LAD", "LCX", "RCA"]
    for v in vessels:
        g = [r for r in done if r["vessel"].upper() == v]
        if not g:
            continue
        def m(k):
            return sum(float(x[k]) for x in g) / len(g)
        print(f"{v:8}{len(g):>4}{m('rel_mse'):>10.4f}{m('rel_rmse'):>11.4f}"
              f"{m('rel_bias'):>10.4f}{m('rel_mae'):>9.4f}{m('rel_p95'):>9.4f}")
    if done:
        def ma(k):
            return sum(float(x[k]) for x in done) / len(done)
        print(f"{'ALL':8}{len(done):>4}{ma('rel_mse'):>10.4f}"
              f"{ma('rel_rmse'):>11.4f}{ma('rel_bias'):>10.4f}"
              f"{ma('rel_mae'):>9.4f}{ma('rel_p95'):>9.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Относительный радиус предсказания vs GT")
    ap.add_argument("--cases", default=str(common.BASE / "cases_120_50.txt"))
    ap.add_argument("--out", default=str(
        common.OUT / "reports" / "metrics_radius_relative.csv"))
    ap.add_argument("--out-mean", default=str(
        common.OUT / "reports" / "metrics_radius_relative_mean.csv"))
    ap.add_argument("--max-near-mm", type=float, default=3.0,
                    help="учитывать точки, где r_pred <= порога (как в metrics.py)")
    ap.add_argument("--min-r-gt", type=float, default=0.0,
                    help="не учитывать точки с r_gt меньше порога (мм)")
    args = ap.parse_args()

    cases = read_cases(Path(args.cases))
    print(f"кейсов: {len(cases)}  (max_near_mm={args.max_near_mm}, "
          f"min_r_gt={args.min_r_gt})")

    per_case = []
    for i, (scan, vessel) in enumerate(cases, 1):
        res = relative_radius(scan, vessel, args.max_near_mm, args.min_r_gt)
        row = {"scan": scan, "vessel": vessel, **res}
        per_case.append(row)
        if res.get("status") != "done" and res.get("status") != "no_mask":
            print(f"  [{i}/{len(cases)}] {scan}_{vessel}: {res.get('status')}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CASE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in per_case:
            w.writerow({k: (round(r[k], 6) if isinstance(r.get(k), float)
                            else r.get(k)) for k in CASE_COLUMNS})

    mean_rows = summarize(per_case)
    mean_path = Path(args.out_mean)
    with open(mean_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "metric", "mean", "std"])
        w.writeheader()
        w.writerows(mean_rows)

    n_done = sum(1 for r in per_case if r.get("status") == "done")
    print(f"\nготово: {n_done}/{len(cases)}")
    print_table(per_case)
    print(f"\nпо кейсам: {out_path}\nсводка   : {mean_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
