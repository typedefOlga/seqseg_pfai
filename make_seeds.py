"""Сид-точки SeqSeg для фрагмента pfai_gen (два направления).

Для сосуда одного скана строятся два кейса:
  <scan>_<vessel>_f — старт в проксимальном клике, трассировка дистально;
  <scan>_<vessel>_b — конец в дистальном клике, трассировка проксимально.

SeqSeg хранит сиды как [old, new, r] в LPS; направление трассировки = new - old.
Клик в RAS переводится в LPS ([-x,-y,z]). `old` откладывается назад вдоль
касательной центрлинии на `--step-mm`.

Запуск:
    python make_seeds.py --scan 908 --vessel lcx
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import common


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def build_seed(sample: dict, direction: str, step_mm: float,
               min_start_radius: float, r_min: float, r_max: float):
    """Возвращает (case, old_lps, new_lps, radius, info) для направления f|b."""
    cl = sample["centerline"]
    p0, p1 = sample["points"]
    xyz = np.array([c["xyz"] for c in cl], dtype=float)
    rad = np.array([c["radius_mm"] for c in cl], dtype=float)

    if direction == "f":
        new = np.asarray(p0["click_xyz"], dtype=float)
        radius_raw = float(p0["radius_mm"])
        tangent = _unit(xyz[1] - xyz[0])          # dir роста arc (дистально)
        old = new - step_mm * tangent
        note = "start click, trace distal"
    elif direction == "b":
        radius_raw = float(p1["radius_mm"])
        if radius_raw >= min_start_radius:
            new = np.asarray(p1["click_xyz"], dtype=float)
            tangent = _unit(xyz[-2] - xyz[-1])    # dir к старту (проксимально)
            note = "end click, trace proximal"
        else:
            k = next((i for i in range(len(cl) - 1, -1, -1)
                      if rad[i] >= min_start_radius), None)
            if k is None:
                # во всём сосуде нет точки шире порога — берём самую широкую,
                # но трассируем по-прежнему проксимально
                k = int(np.argmax(rad))
                new = xyz[k].copy()
                tangent = _unit(xyz[k - 1] - xyz[k])
                note = (f"end click r={radius_raw:.2f}, нет точки >= "
                        f"{min_start_radius:.2f} (max {rad.max():.2f}) -> "
                        f"самая широкая cl[{k}] r={rad[k]:.2f}")
                radius_raw = float(rad[k])
            else:
                new = xyz[k].copy()
                tangent = _unit(xyz[k - 1] - xyz[k])
                note = (f"end click r={radius_raw:.2f}<{min_start_radius:.2f}, "
                        f"inset to cl[{k}] r={rad[k]:.2f}")
                radius_raw = float(rad[k])
        old = new + step_mm * tangent
    else:
        raise ValueError(direction)

    radius = float(np.clip(radius_raw, r_min, r_max))
    old_lps = common.ras_to_lps(old)
    new_lps = common.ras_to_lps(new)
    info = {
        "direction": direction,
        "new_ras": np.round(new, 4).tolist(),
        "old_ras": np.round(old, 4).tolist(),
        "new_lps": np.round(new_lps, 4).tolist(),
        "old_lps": np.round(old_lps, 4).tolist(),
        "tangent_ras": np.round(tangent, 4).tolist(),
        "radius_raw_mm": radius_raw,
        "radius_mm": radius,
        "note": note,
    }
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description="Сиды SeqSeg для фрагмента pfai_gen")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--vessel", required=True)
    ap.add_argument("--step-mm", type=float, default=1.0,
                    help="смещение old назад вдоль касательной")
    ap.add_argument("--min-start-radius", type=float, default=0.8,
                    help="концевой клик тоньше этого радиуса -> отступ внутрь")
    ap.add_argument("--r-min", type=float, default=1.0)
    ap.add_argument("--r-max", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import nibabel as nib

    scan, vessel = args.scan, args.vessel
    sample = common.load_sample(scan, vessel)
    affine = nib.load(str(common.ct_path(scan))).affine
    frag, _ = common.load_mask(common.fragment_path(scan, vessel))

    out_dir = common.OUT / "seeds"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds, mapping = [], []
    print(f"скан {scan} · {vessel.upper()} · длина {sample['length_mm']:.1f} мм")
    print(f"{'case':<16}{'r_raw':>7}{'r':>6}  {'new LPS':<28} note")
    for direction in ("f", "b"):
        info = build_seed(sample, direction, args.step_mm,
                          args.min_start_radius, args.r_min, args.r_max)
        case = common.case_name(scan, vessel, direction)

        # проверка попадания new в маску фрагмента
        vox = common.voxel_index(affine, info["new_ras"])
        in_frag = (all(0 <= vox[i] < frag.shape[i] for i in range(3))
                   and bool(frag[tuple(vox)]))
        info["new_in_fragment"] = in_frag
        info["new_voxel_xyz"] = vox.tolist()

        seeds.append({
            "name": case,
            "seeds": [[info["old_lps"], info["new_lps"], info["radius_mm"]]],
            "cardiac_mesh": False,
        })
        mapping.append({
            "case": case, "scan": scan, "vessel": vessel.upper(),
            **info,
        })
        print(f"{case:<16}{info['radius_raw_mm']:>7.3f}{info['radius_mm']:>6.2f}"
              f"  {str(np.round(info['new_lps'], 3).tolist()):<28}"
              f" {info['note']}  in_frag={in_frag}")

    seeds_path = out_dir / f"seeds_{scan}_{vessel.lower()}.json"
    map_path = out_dir / f"mapping_{scan}_{vessel.lower()}.json"
    seeds_path.write_text(json.dumps(seeds, indent=2), encoding="utf-8")
    map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"\nзаписано: {seeds_path}\n          {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
