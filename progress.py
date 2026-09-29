"""Сводка прогресса батча (read-only).

Читает out/reports/status_shard*.json и runtime_shard*.csv, печатает таблицу
по кейсам и итог: готово/всего, среднее время на кейс, ETA, ошибки.

Запуск:
    python progress.py            # по всем шардам
    python progress.py --shard 0
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import common

RUNNING = {"started", "seqseg_error", "eval_error", "error"}


def elapsed_since(ts: str | None) -> float:
    if not ts:
        return float("nan")
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return float("nan")
    return time.time() - t


def main() -> int:
    ap = argparse.ArgumentParser(description="Прогресс батча")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--reports", default=None)
    ap.add_argument("--cases", default=None,
                    help="файл случаев (для оценки общего числа); "
                         "по умолчанию cases_120.txt")
    args = ap.parse_args()

    reports = Path(args.reports) if args.reports else common.OUT / "reports"
    pattern = (f"status_shard{args.shard}.json" if args.shard is not None
               else "status_*.json")
    files = sorted(glob.glob(str(reports / pattern)))
    if not files:
        print(f"нет статус-файлов в {reports} (батч не запускался?)")
        return 0

    def score(st: dict) -> int:
        if st.get("status") == "done" and not st.get("error"):
            return 3
        if st.get("status") == "started" and not st.get("error"):
            return 2
        return 1

    def event_epoch(st: dict) -> float:
        ts = st.get("finished_at") or st.get("started_at")
        if not ts:
            return 0.0
        try:
            return time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            return 0.0

    # для дубликатов (шард + повторный проход) берём запись с самой свежей
    # временной меткой; при равенстве — с более высоким статусом
    best: dict = {}
    best_key: dict = {}
    for fp in files:
        data = json.loads(Path(fp).read_text(encoding="utf-8"))
        for case, st in data.items():
            key = (event_epoch(st), score(st))
            if case not in best or key >= best_key[case]:
                best[case] = st
                best_key[case] = key
    rows = list(best.items())

    errs = [r for r in rows if r[1].get("error")
            or r[1].get("status") in ("seqseg_error", "eval_error", "error")]
    done = [r for r in rows if r[1].get("status") == "done"
            and not r[1].get("error")]
    running = [r for r in rows if r[1].get("status") == "started"
               and not r[1].get("error")]

    def live_steps(case: str) -> str:
        """Шаги текущего запуска SeqSeg для кейса (f/b) из out.txt."""
        parts = []
        for d in ("f", "b"):
            ot = (common.OUT / "seqseg" /
                  f"run3d_fullres_{case}_{d}" / "out.txt")
            if ot.is_file():
                txt = ot.read_text(encoding="utf-8", errors="ignore")
                parts.append(f"{d}={txt.count('*** Step number')}")
        return "/".join(parts)

    print(f"{'case':<12}{'status':<14}{'steps f/b':>10}{'seqseg_s':>9}"
          f"{'eval_s':>8}{'total_s':>9}{'config':>18}  elapsed")
    print("-" * 96)
    for case, st in rows:
        sf = st.get("f")
        sb = st.get("b")
        steps = ("-" if sf is None and sb is None else f"{sf}/{sb}")
        if st.get("status") == "started":
            live = live_steps(case)
            if live:
                steps = live
        el = (f"{elapsed_since(st.get('started_at')):.0f}s"
              if st.get("status") == "started" else "")
        print(f"{case:<12}{st.get('status', '?'):<14}{steps:>10}"
              f"{str(st.get('seqseg_s', '-')):>9}{str(st.get('eval_s', '-')):>8}"
              f"{str(st.get('total_s', '-')):>9}"
              f"{str(st.get('config', '-')):>18}  {el}")

    cases_path = Path(args.cases) if args.cases \
        else common.BASE / "cases_120.txt"
    expected = None
    if cases_path.is_file():
        expected = len([ln for ln in cases_path.read_text().splitlines()
                        if ln.strip()])
    n_done = len(done)
    n_running = len(running)
    # кэшированные кейсы (готовы ранее) не берём в оценку времени
    fresh = [r[1]["total_s"] for r in done
             if not r[1].get("cached") and "total_s" in r[1]
             and r[1]["total_s"] >= 20.0]
    mean = sum(fresh) / len(fresh) if fresh else float("nan")
    print("-" * 96)
    denom = expected if expected else len(rows)
    print(f"готово: {n_done}/{denom}   в работе: {n_running}   "
          f"ошибок: {len(errs)}")
    if fresh:
        print(f"среднее время на кейс: {mean:.1f}s (n={len(fresh)})")
        remaining = denom - n_done
        eta = remaining * mean / 2.0
        print(f"ETA (2 GPU): ~{eta/60:.0f} мин (осталось {remaining} кейсов)")
    else:
        print("среднее время на кейс: пока нет полных кейсов "
              "(первые ~150 с/артерия)")
    if running:
        for case, st in running:
            print(f"  в работе: {case}  ({elapsed_since(st.get('started_at')):.0f}s)")
    if errs:
        for case, st in errs:
            print(f"  ошибка: {case}: {st.get('error', '?')}")

    # текущие средние по общему файлу метрик
    mp = reports / "metrics_union.csv"
    if mp.is_file():
        import csv
        with open(mp, encoding="utf-8") as f:
            mrows = [r for r in csv.DictReader(f) if r.get("status") == "done"]
        if mrows:
            def mean(c):
                v = [float(r[c]) for r in mrows
                     if r.get(c) not in (None, "", "None")]
                return sum(v) / len(v) if v else float("nan")
            print(f"\nметрик-строк: {len(mrows)}  "
                  f"mean dice={mean('dice'):.3f}  clDice={mean('cl_dice'):.3f}  "
                  f"HD95={mean('hd95'):.2f}  radius_mae={mean('radius_mae'):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
