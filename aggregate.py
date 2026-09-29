"""Средние метрики по файлу out/reports/metrics_union.csv.

Можно запускать в любой момент: считает mean±std по уже обработанным фрагментам
(overall и по артериям LAD/LCx/RCA). Пишет out/reports/metrics_mean.csv
(long-формат: group,n,metric,mean,std) и печатает ключевые метрики.

Запуск:
    python aggregate.py
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

import common

NON_NUMERIC = {"scan", "vessel", "status", "ts", "config"}
KEY = ["dice", "cl_dice", "hd", "hd95", "assd", "chamfer",
       "boundary_f1@0.12", "b1_err", "centerline_md", "centerline_hd95",
       "coverage@1.0mm", "radius_mae", "total_s", "seqseg_s", "eval_s"]


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f)]


def numeric_cols(rows: list[dict]) -> list[str]:
    cols = []
    for c in rows[0].keys():
        if c in NON_NUMERIC:
            continue
        ok = False
        for r in rows:
            try:
                float(r[c]); ok = True; break
            except (TypeError, ValueError):
                continue
        if ok:
            cols.append(c)
    return cols


def summarize(rows: list[dict], by: str | None = None) -> list[dict]:
    cols = numeric_cols(rows)
    groups = {"all": rows}
    if by:
        groups = {}
        for r in rows:
            groups.setdefault(r[by].upper(), []).append(r)
    out = []
    for gname, grp in groups.items():
        for c in cols:
            vals = [float(r[c]) for r in grp
                    if r.get(c) not in (None, "", "None")]
            if not vals:
                continue
            out.append({"group": gname, "n": len(vals), "metric": c,
                        "mean": round(sum(vals) / len(vals), 5),
                        "std": round(st.pstdev(vals), 5) if len(vals) > 1
                        else 0.0})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Средние метрики")
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = Path(args.metrics) if args.metrics \
        else common.OUT / "reports" / "metrics_union.csv"
    rows = [r for r in load_rows(src) if r.get("status") == "done"]
    if not rows:
        print(f"нет готовых строк в {src}")
        return 0

    all_sum = summarize(rows)
    per_v = summarize(rows, by="vessel")

    dst = Path(args.out) if args.out else src.parent / "metrics_mean.csv"
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "n", "metric", "mean", "std"])
        w.writeheader()
        w.writerows(all_sum)
        w.writerows(per_v)

    print(f"фрагментов готово: {len(rows)}  (overall + по артериям)")
    def fmt(group):
        d = {s["metric"]: s for s in per_v + all_sum if s["group"] == group}
        cells = []
        for k in KEY:
            if k in d:
                cells.append(f"{k}={d[k]['mean']:.3f}")
        return "  ".join(cells)
    print("ALL :", fmt("all"))
    for v in sorted({r["vessel"].upper() for r in rows}):
        print(f"{v:4}:", fmt(v))
    print(f"\nзаписано: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
