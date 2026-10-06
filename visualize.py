"""Визуализация масок SeqSeg vs GT в трёх проекциях (как p5/p50/p95).

Схема — clscat из `viz_diagnose.py`: категориально, ровно один класс на
пиксель (приоритет пересечение > pred-only > GT-only), ровно 3 цвета:
    зелёный  — пересечение GT и предсказания;
    красный  — предсказание без GT (FP);
    синий    — GT без предсказания (FN).

Берётся вариант UNION (F ∪ B) из out/eval/<scan>_<vessel>/union.nii.gz.
Выход: out/viz/<scan>_<vessel>_clscat.png (3 проекции).

Запуск:
    python visualize.py --scan 306 --vessel lad
    python visualize.py --scan 306 --vessel lad --scan 908 --vessel lcx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import common
import viz_diagnose


def main() -> int:
    ap = argparse.ArgumentParser(description="clscat-визуализация SeqSeg vs GT")
    ap.add_argument("--scan", action="append", required=True)
    ap.add_argument("--vessel", action="append", required=True)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--margin-mm", type=float, default=20.0)
    ap.add_argument("--pred", choices=("union", "cropped", "f", "b", "inter"),
                    default="union",
                    help="какую маску брать: union/inter/f/b (eval) или cropped (recount)")
    ap.add_argument("--full", action="store_true",
                    help="полное поле (всё сердце), без кропа по GT")
    ap.add_argument("--centerline", action="store_true",
                    help="рисовать центрлинию GT поверх")
    args = ap.parse_args()

    if len(args.scan) != len(args.vessel):
        print("число --scan и --vessel должно совпадать")
        return 1
    viz_dir = (Path(args.outdir) if args.outdir else common.OUT / "viz")
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for scan, vessel in zip(args.scan, args.vessel):
        if args.pred == "cropped":
            pred_path = (common.OUT / "recount" /
                         f"{scan}_{vessel}_cropped.nii.gz")
            tag = "cropped_clscat"
        elif args.pred == "union":
            pred_path, tag = None, "clscat"
        else:
            pred_path = (common.OUT / "eval" /
                         f"{scan}_{vessel}" / f"{args.pred}.nii.gz")
            tag = f"{args.pred}_clscat"
        n += viz_diagnose.render(scan, vessel, viz_dir, args.margin_mm,
                                 pred_path, tag, args.full, args.centerline)
    print(f"\nготово: {n} фигур в {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
