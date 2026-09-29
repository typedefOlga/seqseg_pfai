"""Фоновый resumable-драйвер батча SeqSeg по фрагментам pfai_gen.

Для каждого кейса (scan + vessel): make_seeds -> seqseg run batch (оба конца)
-> evaluate. Пропускает уже готовые шаги. После каждого кейса пишет статус и
строку тайминга в out/reports/.

Примеры:
    # один скан (все три артерии), одна GPU
    python run_batch.py --only-scan 100 --gpu 0

    # шард батча: 2 процесса, каждый на своей GPU
    python run_batch.py --shard 0/2 --gpu 0 --delete-mha
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

import common

PY = sys.executable
MAX_VOXELS = 200_000_000


def read_cases(path: Path) -> list[tuple[str, str]]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        scan, vessel = line.split()
        cases.append((scan, vessel.lower()))
    return cases


def expected_voxels(scan: str, cent: float) -> float:
    img = nib.load(str(common.ct_path(scan)))
    size = np.array(img.shape, dtype=float)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    return float(np.prod(size * spacing / cent))


def pick_config(scan: str) -> str:
    if expected_voxels(scan, 0.3) <= MAX_VOXELS:
        return "global_coro"
    return "global_coro_cent0.4"


def mask_paths(scan: str, vessel: str) -> dict:
    pat = str(common.OUT / "seqseg" / "run" /
              f"{scan}_{vessel}_*_segmentation_*.mha")
    found = {}
    for p in glob.glob(pat):
        m = re.search(rf"{scan}_{vessel}_(f|b)_segmentation_", Path(p).name)
        if m:
            found[m.group(1)] = p
    return found


def steps_for(scan: str, vessel: str, direction: str):
    ot = (common.OUT / "seqseg" /
          f"run3d_fullres_{scan}_{vessel}_{direction}" / "out.txt")
    if not ot.is_file():
        return None
    txt = ot.read_text(encoding="utf-8", errors="ignore")
    return len(re.findall(r"\*\*\* Step number", txt))


def eval_done(scan: str, vessel: str, union_only: bool = False) -> bool:
    d = common.OUT / "eval" / f"{scan}_{vessel}"
    if union_only:
        return (d / "union.json").is_file()
    return all((d / f"{v}.json").is_file()
               for v in ("f", "b", "inter", "union"))


def metrics_has(path, scan: str, vessel: str) -> bool:
    """Есть ли уже строка этого кейса в общем файле метрик."""
    p = Path(path)
    if not p.is_file():
        return False
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{scan},{vessel},"):
                return True
    return False


def run(cmd: list[str], log: Path) -> int:
    with open(log, "a", encoding="utf-8") as lf:
        lf.write(f"\n$ {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                              cwd=str(common.BASE))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="Батч SeqSeg по фрагментам")
    ap.add_argument("--cases", default=str(common.BASE / "cases.txt"))
    ap.add_argument("--shard", default="0/1", help="k/N")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--only-scan", default=None)
    ap.add_argument("--only-vessel", default=None)
    ap.add_argument("--max-n-steps", type=int, default=200)
    ap.add_argument("--min-start-radius", type=float, default=1.2)
    ap.add_argument("--delete-mha", action="store_true")
    ap.add_argument("--union-only", action="store_true")
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--metrics-file", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--run-id", default=None,
                    help="суффикс имён status/runtime/log (по умолчанию shard<k>)")
    args = ap.parse_args()
    metrics_file = (args.metrics_file
                    or str(common.OUT / "reports" / "metrics_union.csv"))

    k, n = (int(x) for x in args.shard.split("/"))
    all_cases = read_cases(Path(args.cases))
    if args.only_scan:
        all_cases = [c for c in all_cases if c[0] == args.only_scan]
    if args.only_vessel:
        all_cases = [c for c in all_cases if c[1] == args.only_vessel]
    cases = all_cases[k::n]

    reports = (Path(args.reports) if args.reports
               else common.OUT / "reports")
    reports.mkdir(parents=True, exist_ok=True)
    rid = args.run_id or f"shard{k}"
    status_path = reports / f"status_{rid}.json"
    runtime_path = reports / f"runtime_{rid}.csv"
    log = reports / f"batch_{rid}.log"

    status = json.loads(status_path.read_text(encoding="utf-8")) \
        if status_path.is_file() else {}
    if not runtime_path.is_file():
        runtime_path.write_text(
            "scan,vessel,status,steps_f,steps_b,seqseg_s,eval_s,total_s,config\n",
            encoding="utf-8")

    def save_status():
        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False),
                               encoding="utf-8")

    print(f"shard {k}/{n}: {len(cases)} кейсов, GPU {args.gpu}", flush=True)
    for i, (scan, vessel) in enumerate(cases, 1):
        case = f"{scan}_{vessel}"
        print(f"\n[{i}/{len(cases)}] {case}", flush=True)
        case_log = reports / f"case_{case}.log"
        st = {"status": "started",
              "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        status[case] = st
        save_status()
        t0 = time.time()
        steps = {}
        seeds_s = 0.0
        already_eval = eval_done(scan, vessel, args.union_only)
        have_metrics = metrics_has(metrics_file, scan, vessel)
        try:
            if already_eval and have_metrics:
                st = {"status": "done", "cached": True}
                status[case] = st
                save_status()
                print(f"  уже готово (eval+метрики есть)", flush=True)
                continue

            if not already_eval:
                # --- seeds
                seeds = common.OUT / "seeds" / f"seeds_{scan}_{vessel}.json"
                if not seeds.is_file():
                    t = time.time()
                    rc = run([PY, "make_seeds.py", "--scan", scan, "--vessel",
                              vessel, "--min-start-radius",
                              str(args.min_start_radius)], case_log)
                    if rc != 0:
                        raise RuntimeError(f"make_seeds rc={rc}")
                    seeds_s = round(time.time() - t, 1)
                    st["seeds_s"] = seeds_s

                # --- seqseg
                have = mask_paths(scan, vessel)
                if not ("f" in have and "b" in have):
                    cfg = pick_config(scan)
                    st["config"] = cfg
                    t = time.time()
                    rc = run([PY, "run_seqseg.py", "--scan", scan, "--vessel",
                              vessel, "--direction", "both", "--gpu",
                              str(args.gpu), "--config-name", cfg,
                              "--max-n-steps", str(args.max_n_steps)], case_log)
                    st["seqseg_s"] = round(time.time() - t, 1)
                    steps["f"] = steps_for(scan, vessel, "f")
                    steps["b"] = steps_for(scan, vessel, "b")
                    if rc != 0:
                        st.update(status="seqseg_error", rc=rc, **steps)
                        status[case] = st
                        save_status()
                        raise RuntimeError(f"seqseg rc={rc}")

            # --- evaluate
            if not (already_eval and have_metrics):
                sf = steps.get("f") if steps.get("f") is not None else -1
                sb = steps.get("b") if steps.get("b") is not None else -1
                cmd = [PY, "evaluate.py", "--scan", scan, "--vessel", vessel,
                       "--metrics-file", metrics_file,
                       "--steps-f", str(sf),
                       "--steps-b", str(sb),
                       "--seeds-s", str(seeds_s),
                       "--seqseg-s", str(st.get("seqseg_s") or 0.0),
                       "--config", str(st.get("config") or "")]
                if args.union_only:
                    cmd.append("--union-only")
                if args.crop:
                    cmd.append("--crop")
                t = time.time()
                rc = run(cmd, case_log)
                st["eval_s"] = round(time.time() - t, 1)
                if rc != 0:
                    st.update(status="eval_error", rc=rc)
                    status[case] = st
                    save_status()
                    raise RuntimeError(f"evaluate rc={rc}")

            # --- cleanup
            if args.delete_mha:
                for p in glob.glob(str(common.OUT / "seqseg" / "run" /
                                        f"{scan}_{vessel}_*.mha")):
                    Path(p).unlink(missing_ok=True)
                for p in glob.glob(str(common.OUT / "seqseg" / "run" /
                                        f"{scan}_{vessel}_*.vtp")):
                    Path(p).unlink(missing_ok=True)
                for d in (common.OUT / "seqseg" /
                          f"run3d_fullres_{scan}_{vessel}_f",
                          common.OUT / "seqseg" /
                          f"run3d_fullres_{scan}_{vessel}_b"):
                    if d.is_dir():
                        for f in d.rglob("*"):
                            if f.is_file():
                                f.unlink()
                        for f in sorted(d.rglob("*"), reverse=True):
                            if f.is_dir():
                                f.rmdir()
                        d.rmdir()

            total = round(time.time() - t0, 1)
            st.update(status="done", total_s=total, cached=already_eval,
                      finished_at=time.strftime("%Y-%m-%d %H:%M:%S"), **steps)
            status[case] = st
            save_status()
            with open(runtime_path, "a", encoding="utf-8") as f:
                f.write(f"{scan},{vessel},done,{steps.get('f')},"
                        f"{steps.get('b')},{st.get('seqseg_s')},"
                        f"{st.get('eval_s')},{total},{st.get('config')}\n")
            print(f"  готово: total={total}s seqseg={st.get('seqseg_s')}s "
                  f"steps f/b={steps.get('f')}/{steps.get('b')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            if st.get("status") not in ("seqseg_error", "eval_error"):
                st["status"] = "error"
            st["error"] = str(exc)
            st["total_s"] = round(time.time() - t0, 1)
            status[case] = st
            save_status()
            with open(runtime_path, "a", encoding="utf-8") as f:
                f.write(f"{scan},{vessel},{st['status']},{steps.get('f')},"
                        f"{steps.get('b')},{st.get('seqseg_s')},"
                        f"{st.get('eval_s')},{st['total_s']},"
                        f"{st.get('config')}\n")
            print(f"  ОШИБКА: {exc}", flush=True)

    print("\nшард завершён", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
