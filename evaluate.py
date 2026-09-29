"""Оценка масок SeqSeg против истинной маски фрагмента pfai_gen.

По умолчанию: ресемплит выходные .mha SeqSeg на нативную сетку КТ, строит
варианты F (старт), B (конец), F∩B, F∪B и считает метрики (metrics.py).

Режим --union-only: считается только UNION = F ∪ B (если одно направление
отсутствует — берётся доступное).
Режим --crop: метрики считаются в bbox(pred ∪ gt) (точный и быстрый).

Запуск:
    python evaluate.py --scan 908 --vessel lcx
    python evaluate.py --scan 100 --vessel lad --union-only --crop --metrics-file out/reports/metrics_union.csv
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import glob
import json
import os
import re
import time
from pathlib import Path

import nibabel as nib
import numpy as np

import common
import metrics

METRIC_COLUMNS = [
    "scan", "vessel", "status", "ts", "steps_f", "steps_b",
    "seqseg_s", "eval_s", "total_s", "config",
    "dice", "cl_dice", "precision", "recall", "f1",
    "hd", "hd95", "assd", "chamfer",
    "boundary_f1@0.12", "boundary_precision@0.12", "boundary_recall@0.12",
    "b0_pred", "b1_pred", "b2_pred", "b0_gt", "b1_gt", "b2_gt",
    "b0_err", "b1_err", "b2_err",
    "centerline_md", "centerline_hd95", "coverage@1.0mm",
    "radius_mae", "radius_rmse", "radius_bias",
    "pred_voxels", "gt_voxels",
]


def discover_masks(scan: str, vessel: str) -> dict:
    pattern = str(common.OUT / "seqseg" / "run" /
                  f"{scan}_{vessel.lower()}_*_segmentation_*_steps.mha")
    found = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(rf"{scan}_{vessel.lower()}_(f|b)_segmentation_",
                      Path(path).name)
        if m:
            found[m.group(1)] = path
    return found


def crop_to_union(pred: np.ndarray, gt: np.ndarray, affine: np.ndarray):
    """bbox(pred ∪ gt), нулевой margin. Возвращает (pred_c, gt_c, affine_c)."""
    u = pred | gt
    idx = np.argwhere(u)
    if idx.size == 0:
        return pred, gt, affine
    lo = idx.min(0)
    hi = idx.max(0) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
    aff_c = affine.copy()
    aff_c[:3, 3] = affine[:3, 3] + affine[:3, :3] @ lo[::-1]  # lo (z,y,x)->(x,y,z)
    return pred[sl], gt[sl], aff_c


def append_metrics(path: Path, row: dict) -> None:
    """Добавляет строку метрик в общий CSV под flock (один воркер за раз)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            write_header = (f.tell() == 0)
            w = csv.DictWriter(f, fieldnames=METRIC_COLUMNS,
                               extrasaction="ignore")
            if write_header:
                w.writeheader()
            clean = {k: (round(row[k], 6) if isinstance(row.get(k), float)
                         else row.get(k)) for k in METRIC_COLUMNS}
            w.writerow(clean)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main() -> int:
    ap = argparse.ArgumentParser(description="Метрики SeqSeg vs фрагмент")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--vessel", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--union-only", action="store_true")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--metrics-file", default=None)
    ap.add_argument("--steps-f", type=int, default=None)
    ap.add_argument("--steps-b", type=int, default=None)
    ap.add_argument("--seeds-s", type=float, default=None)
    ap.add_argument("--seqseg-s", type=float, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    scan, vessel = args.scan, args.vessel
    t_eval = time.time()

    sample = common.load_sample(scan, vessel)
    ct_img = nib.load(str(common.ct_path(scan)))
    affine = ct_img.affine
    spacing_zyx = tuple(float(z) for z in ct_img.header.get_zooms()[:3][::-1])

    gt_xyz, _ = common.load_mask(common.fragment_path(scan, vessel))
    gt = np.transpose(gt_xyz, (2, 1, 0))          # (z, y, x)
    gt_cl_world = np.array([c["xyz"] for c in sample["centerline"]], float)
    gt_radius = np.array([c["radius_mm"] for c in sample["centerline"]], float)

    out_dir = (Path(args.outdir) if args.outdir
               else common.OUT / "eval" / f"{scan}_{vessel.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = discover_masks(scan, vessel)
    if masks:
        preds = {d.upper(): common.resample_to_ct(p, scan)
                 for d, p in masks.items()}
    else:
        # fallback: уже ресемплированные маски в out/eval/<case>/{f,b}.nii.gz
        preds = {}
        for d in ("f", "b"):
            p = out_dir / f"{d}.nii.gz"
            if p.is_file():
                arr = np.asanyarray(nib.load(str(p)).dataobj) > 0
                preds[d.upper()] = np.transpose(arr, (2, 1, 0))
        if preds:
            print(f"использую сохранённые маски из {out_dir}")

    # Отбрасываем заведомо битые направления: при провале старта SeqSeg
    # записывает маску из одних единиц (весь том). Реальные сосуды < ~5%.
    for d in list(preds):
        frac = float(preds[d].mean())
        if frac > 0.5:
            print(f"  направление {d}: маска покрывает {frac*100:.1f}% "
                  f"объёма — отбрасываю как битую")
            preds.pop(d)

    if not preds:
        print(f"нет масок SeqSeg под шаблон out/seqseg/run/{scan}_{vessel}_*")
        if args.metrics_file:
            append_metrics(Path(args.metrics_file), {
                "scan": scan, "vessel": vessel, "status": "no_mask",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "steps_f": args.steps_f, "steps_b": args.steps_b,
                "seqseg_s": args.seqseg_s, "config": args.config})
        return 1
    if args.union_only:
        if not preds:
            variants = {}
        elif len(preds) == 1:
            variants = {"UNION": next(iter(preds.values()))}
        else:
            variants = {"UNION": preds["F"] | preds["B"]}
    else:
        variants = dict(preds)
        if "F" in variants and "B" in variants:
            variants["INTER"] = variants["F"] & variants["B"]
            variants["UNION"] = variants["F"] | variants["B"]

    rows = []
    for name, pred in variants.items():
        nib.save(nib.Nifti1Image(
            np.transpose(pred, (2, 1, 0)).astype(np.uint8), affine),
            str(out_dir / f"{name.lower()}.nii.gz"))

        if args.crop:
            pc, gc, aff = crop_to_union(pred, gt, affine)
        else:
            pc, gc, aff = pred, gt, affine

        res = metrics.evaluate_mask(pc, gc, spacing_zyx,
                                    gt_cl_world, gt_radius, aff)
        res["variant"] = name
        rows.append(res)
        (out_dir / f"{name.lower()}.json").write_text(
            json.dumps(res, indent=2, default=float), encoding="utf-8")

    keys = ["variant"] + [c for c in METRIC_COLUMNS
                          if c not in ("scan", "vessel", "status", "ts")]
    csv_path = out_dir.parent / f"summary_{scan}_{vessel.lower()}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(row[k], 4) if isinstance(row.get(k), float)
                            else row.get(k)) for k in keys})

    eval_s = round(time.time() - t_eval, 1)
    total_s = None
    if args.seqseg_s is not None:
        total_s = round((args.seeds_s or 0.0) + (args.seqseg_s or 0.0)
                        + eval_s, 1)
    if args.metrics_file:
        for r in rows:
            rec = dict(r)
            rec.update({"scan": scan, "vessel": vessel, "status": "done",
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "steps_f": args.steps_f, "steps_b": args.steps_b,
                        "seqseg_s": args.seqseg_s, "eval_s": eval_s,
                        "total_s": total_s, "config": args.config})
            append_metrics(Path(args.metrics_file), rec)

    hdr = f"{'variant':<7}{'dice':>7}{'cldice':>8}{'hd':>8}{'hd95':>8}" \
          f"{'chamfer':>9}{'bF1@.12':>9}{'b0err':>7}{'b1err':>7}{'b2err':>7}" \
          f"{'cl_md':>7}{'cl95':>7}{'cov':>6}{'r_mae':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:<7}{r['dice']:>7.3f}{r['cl_dice']:>8.3f}"
              f"{r['hd']:>8.2f}{r['hd95']:>8.2f}{r['chamfer']:>9.3f}"
              f"{r['boundary_f1@0.12']:>9.3f}{r['b0_err']:>7}{r['b1_err']:>7}"
              f"{r['b2_err']:>7}{r.get('centerline_md', float('nan')):>7.2f}"
              f"{r.get('centerline_hd95', float('nan')):>7.2f}"
              f"{r.get('coverage@1.0mm', float('nan')):>6.2f}"
              f"{r.get('radius_mae', float('nan')):>7.3f}")
    print(f"\nмаски: {out_dir}\nсводка: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
