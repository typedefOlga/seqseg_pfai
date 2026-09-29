"""Запуск SeqSeg на кейсах фрагмента (два направления).

Строит staging-каталог (images/ с симлинками на КТ + seeds.json) и вызывает
`seqseg run batch`. По умолчанию один запуск обрабатывает оба кейса (<case>_f,
<case>_b), загружая веса nnU-Net один раз.

Запуск (только на полностью свободной GPU):
    python run_seqseg.py --scan 908 --vessel lcx --gpu 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import common


def build_staging(scan: str, vessel: str, staging: Path) -> list[str]:
    """images/<case>.nii.gz -> КТ (симлинки) + seeds.json. Возвращает имена кейсов."""
    seeds_src = common.OUT / "seeds" / f"seeds_{scan}_{vessel.lower()}.json"
    seeds = json.loads(seeds_src.read_text(encoding="utf-8"))

    images = staging / "images"
    if staging.exists():
        shutil.rmtree(staging)
    images.mkdir(parents=True)

    cases = []
    for entry in seeds:
        case = entry["name"]
        link = images / f"{case}.nii.gz"
        os.symlink(str(common.ct_path(scan)), link)
        cases.append(case)
    (staging / "seeds.json").write_text(
        json.dumps(seeds, indent=2), encoding="utf-8")
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="SeqSeg batch для фрагмента")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--vessel", required=True)
    ap.add_argument("--gpu", type=int, default=0,
                    help="индекс CUDA_VISIBLE_DEVICES (должна быть свободна)")
    ap.add_argument("--config-name", default="global_coro")
    ap.add_argument("--max-n-steps", type=int, default=200)
    ap.add_argument("--max-n-steps-per-branch", type=int, default=100)
    ap.add_argument("--max-n-branches", type=int, default=20)
    ap.add_argument("--fold", default="all")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--unit", default="mm")
    ap.add_argument("--direction", choices=("f", "b", "both"), default="both",
                    help="какой конец запускать (f — старт, b — конец)")
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--staging", default=None)
    args = ap.parse_args()

    if args.start is None or args.stop is None:
        edges = {"f": (0, 1), "b": (1, 2), "both": (0, -1)}
        args.start, args.stop = edges[args.direction]

    scan, vessel = args.scan, args.vessel
    staging = (Path(args.staging) if args.staging
               else common.OUT / "seqseg" / f"staging_{scan}_{vessel.lower()}")
    outdir = (Path(args.outdir) if args.outdir
              else common.OUT / "seqseg" / "run")
    outdir.mkdir(parents=True, exist_ok=True)

    cases = build_staging(scan, vessel, staging)
    print(f"staging: {staging}")
    print(f"кейсы  : {cases}")
    print(f"GPU    : CUDA_VISIBLE_DEVICES={args.gpu}")

    cmd = [
        str(common.SEQSEG_BIN), "run", "batch",
        "-data_dir", str(staging),
        "-img_ext", ".nii.gz",
        "-outdir", str(outdir),
        "-nnunet_results_path", str(common.WEIGHTS_ROOT),
        "-train_dataset", common.TRAIN_DATASET,
        "-nnunet_type", common.NNUNET_TYPE,
        "-fold", args.fold,
        "-config_name", args.config_name,
        "-scale", str(args.scale),
        "-unit", args.unit,
        "-max_n_steps", str(args.max_n_steps),
        "-max_n_steps_per_branch", str(args.max_n_steps_per_branch),
        "-max_n_branches", str(args.max_n_branches),
        "-start", str(args.start),
        "-stop", str(args.stop),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PATH"] = f"{common.SEQSEG_BIN.parent}:{env.get('PATH', '')}"

    record = {"scan": scan, "vessel": vessel, "gpu": args.gpu, "cases": cases,
              "cmd": cmd, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("$", " ".join(cmd), flush=True)
    t0 = time.time()
    log = common.OUT / "seqseg" / f"log_{scan}_{vessel.lower()}.txt"
    chunks = []
    proc = subprocess.Popen(cmd, env=env, cwd=str(common.BASE),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    with open(log, "w", encoding="utf-8") as lf:
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            chunks.append(line)
            if "*** Step number" in line or "branch" in line.lower():
                print("  " + line.rstrip(), flush=True)
    proc.wait()
    elapsed = time.time() - t0
    record["returncode"] = proc.returncode
    record["elapsed_s"] = round(elapsed, 1)
    record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    record["log"] = str(log)
    (common.OUT / "seqseg" / f"run_{scan}_{vessel.lower()}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    print("".join(chunks)[-2000:], flush=True)
    print(f"\nreturncode={proc.returncode}  время={elapsed/60:.2f} мин  лог={log}",
          flush=True)
    if proc.returncode != 0:
        return proc.returncode

    masks = sorted(glob.glob(str(outdir / f"*_segmentation_*_steps.mha")))
    print("\nмаски:")
    for m in masks:
        print(" ", m)
    record["masks"] = masks
    (common.OUT / "seqseg" / f"run_{scan}_{vessel.lower()}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
