"""Батч-обрезка всех предсказанных масок SeqSeg (union) по двум кликам.

Список кейсов: строки status=done из out/reports/metrics_union.csv (119).
Метод обрезки — как в recount/crop_one.py (полная атрибуция вокселей:
скелет -> полный граф -> путь клик-клик -> оставить воксели, атрибутированные
узлам пути). GT нигде не используется в обрезке.

Метрики считаются ровно как в evaluate.py (переиспользуются функции):
  evaluate.crop_to_union + metrics.evaluate_mask (те же столбцы).

Выход:
  recount/out/<scan>_<vessel>_cropped.nii.gz
  recount/out/metrics_cropped.csv   (колонки как в metrics_union.csv)
  recount/out/crop_summary.csv      (флаги/воксели/путь)

Запуск:
  python recount/crop_all.py
  python recount/crop_all.py --force --limit 3
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

RECOUNT = Path(__file__).resolve().parent
sys.path.insert(0, str(RECOUNT.parent))

import common  # noqa: E402
import evaluate  # noqa: E402
import metrics  # noqa: E402
from crop_one import crop_case  # noqa: E402

SUMMARY_COLUMNS = [
    "scan", "vessel", "status", "ts",
    "n_skel_nodes", "n_path_nodes", "path_arc_mm",
    "proj_dist_start", "proj_dist_end",
    "start_click_in_mask", "end_click_in_mask",
    "union_voxels", "cropped_voxels", "gt_voxels", "removed_frac",
    "warnings",
]


def done_cases(metrics_file: Path) -> list[tuple[str, str]]:
    cases = []
    for r in csv.DictReader(open(metrics_file, encoding="utf-8")):
        if r.get("status") == "done":
            cases.append((r["scan"], r["vessel"]))
    seen, out = set(), []
    for c in cases:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def append_row(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            write_header = (f.tell() == 0)
            w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k) for k in columns})
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_done_metrics(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {(r["scan"], r["vessel"])
            for r in csv.DictReader(open(path, encoding="utf-8"))
            if r.get("status") == "done"}


def process(scan: str, vessel: str, out_dir: Path):
    """Обрезает один кейс, сохраняет .nii.gz, возвращает (info, metric_row)."""
    d = crop_case(scan, vessel)
    info = d["info"]
    affine = d["affine"]
    cropped_xyz = d["cropped_xyz"]

    nii = out_dir / f"{scan}_{vessel}_cropped.nii.gz"
    nib.save(nib.Nifti1Image(cropped_xyz.astype(np.uint8), affine), str(nii))

    # --- метрики ровно как в evaluate.py ---
    sample = common.load_sample(scan, vessel)
    ct_img = nib.load(str(common.ct_path(scan)))
    spacing_zyx = tuple(float(z) for z in ct_img.header.get_zooms()[:3][::-1])
    gt_xyz = common.load_mask(common.fragment_path(scan, vessel))[0]
    gt = np.transpose(gt_xyz, (2, 1, 0))
    gt_cl_world = np.array([c["xyz"] for c in sample["centerline"]], float)
    gt_radius = np.array([c["radius_mm"] for c in sample["centerline"]], float)

    pred = np.transpose(cropped_xyz, (2, 1, 0))       # (z,y,x)
    pc, gc, aff = evaluate.crop_to_union(pred, gt, affine)
    res = metrics.evaluate_mask(pc, gc, spacing_zyx, gt_cl_world, gt_radius, aff)
    res["variant"] = "CROP"

    row = dict(res)
    row.update({
        "scan": scan, "vessel": vessel, "status": "done",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": "crop_all",
    })
    return info, row


def main() -> int:
    ap = argparse.ArgumentParser(description="Батч-обрезка union-масок")
    ap.add_argument("--metrics-file",
                    default=str(common.OUT / "reports" / "metrics_union.csv"))
    ap.add_argument("--outdir", default=str(RECOUNT / "out"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mfile = Path(args.metrics_file)
    metrics_out = out_dir / "metrics_cropped.csv"
    summary_out = out_dir / "crop_summary.csv"

    cases = done_cases(mfile)
    if args.limit:
        cases = cases[:args.limit]
    done = load_done_metrics(metrics_out)
    print(f"кейсов: {len(cases)}; уже с метриками: {len(done)}", flush=True)

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    for i, (scan, vessel) in enumerate(cases, 1):
        nii = out_dir / f"{scan}_{vessel}_cropped.nii.gz"
        if not args.force and nii.is_file() and (scan, vessel) in done:
            n_skip += 1
            continue
        ts = time.strftime("%H:%M:%S")
        try:
            info, row = process(scan, vessel, out_dir)
            append_row(metrics_out, evaluate.METRIC_COLUMNS, row)
            append_row(summary_out, SUMMARY_COLUMNS, {
                "scan": scan, "vessel": vessel, "status": "done",
                "ts": row["ts"],
                "n_skel_nodes": info["n_skel_nodes"],
                "n_path_nodes": info["n_path_nodes"],
                "path_arc_mm": info["path_arc_mm"],
                "proj_dist_start": info["proj_dist_mm"][0],
                "proj_dist_end": info["proj_dist_mm"][1],
                "start_click_in_mask": info["start_click_in_mask"],
                "end_click_in_mask": info["end_click_in_mask"],
                "union_voxels": info["union_voxels"],
                "cropped_voxels": info["cropped_voxels"],
                "gt_voxels": info["gt_voxels"],
                "removed_frac": info["removed_frac"],
                "warnings": "; ".join(info.get("warnings", [])),
            })
            n_ok += 1
            print(f"[{i}/{len(cases)}] {ts} {scan}_{vessel}: union "
                  f"{info['union_voxels']} -> {info['cropped_voxels']} "
                  f"dice={row['dice']:.3f}"
                  + (f" WARN: {info['warnings']}" if info["warnings"] else ""),
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            n_err += 1
            append_row(summary_out, SUMMARY_COLUMNS, {
                "scan": scan, "vessel": vessel, "status": "error",
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "warnings": f"{type(exc).__name__}: {exc}",
            })
            print(f"[{i}/{len(cases)}] {ts} {scan}_{vessel}: ERROR "
                  f"{type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

    dt = time.time() - t0
    print(f"\nготово за {dt/60:.1f} мин: ok={n_ok} skip={n_skip} err={n_err}")
    print(f"маски   : {out_dir}/<scan>_<vessel>_cropped.nii.gz")
    print(f"метрики : {metrics_out}\nсводка  : {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
