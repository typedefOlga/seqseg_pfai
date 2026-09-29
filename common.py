"""Общие утилиты проекта seqseg_pfai.

Пайплайн: фрагмент из pfai_gen (скан + 2 клика -> маска) -> SeqSeg с двух
концов -> пересечение/объединение -> метрики против истинной маски фрагмента
-> визуализация в трёх анатомических проекциях.

Данные только для чтения; все артефакты — в out/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
#  Пути                                                                        #
# --------------------------------------------------------------------------- #

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"

GENERATOR_OUT = Path(
    "/home/y.pchelintsev/vessel-seg/pfai_gen/generator/out/generate")
SAMPLES_DIR = GENERATOR_OUT / "samples"
FRAGMENTS_DIR = GENERATOR_OUT / "fragments"

CT_DIR = Path("/srv/fast1/y.pchelitsev/datasets/ImageCAS/data")

SEQSEG_BIN = Path("/home/o.tonisheva/anaconda3/envs/seqseg/bin/seqseg")
SEQSEG_PY = Path("/home/o.tonisheva/anaconda3/envs/seqseg/bin/python")
WEIGHTS_ROOT = Path(
    "/home/o.tonisheva/oc_seqseg/weights/nnUNet_results_coronary")
TRAIN_DATASET = "Dataset010_SEQCOROASOCACT"
NNUNET_TYPE = "3d_fullres"

# координатная поправка: NIfTI/RAS -> VTK/ITK/LPS (инволюция)
RAS_TO_LPS = np.array([-1.0, -1.0, 1.0])

HU_WINDOW = (-150.0, 600.0)
GREY_GAMMA = 1.0

# Анатомические панели: (заголовок, drop, row, col, метка строк, метка столбцов)
PROJECTION_PANELS = (
    ("Сагиттальная", 0, 2, 1, "S/I (z)", "A/P (y)"),
    ("Корональная", 1, 2, 0, "S/I (z)", "R/L (x)"),
    ("Аксиальная", 2, 1, 0, "A/P (y)", "R/L (x)"),
)


# --------------------------------------------------------------------------- #
#  Пути к данным                                                               #
# --------------------------------------------------------------------------- #

def sample_path(scan_id: str, vessel: str) -> Path:
    return SAMPLES_DIR / f"{scan_id}_{vessel.lower()}.json"


def fragment_path(scan_id: str, vessel: str) -> Path:
    return FRAGMENTS_DIR / f"{scan_id}_{vessel.lower()}.nii.gz"


def ct_path(scan_id: str) -> Path:
    return CT_DIR / f"{scan_id}.img.nii.gz"


def case_name(scan_id: str, vessel: str, direction: str) -> str:
    """Имя кейса SeqSeg: <scan>_<vessel>_<f|b>."""
    return f"{scan_id}_{vessel.lower()}_{direction}"


def split_case(case: str) -> tuple[str, str, str]:
    scan, vessel, direction = case.rsplit("_", 2)
    return scan, vessel, direction


# --------------------------------------------------------------------------- #
#  Координаты                                                                  #
# --------------------------------------------------------------------------- #

def ras_to_lps(points) -> np.ndarray:
    """Мир RAS (NIfTI) -> LPS (SimpleITK/VTK). Знаки x и y меняются."""
    return np.asarray(points, dtype=float) * RAS_TO_LPS


def to_voxel(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Мировые мм -> непрерывные индексы вокселей (обратный affine)."""
    inv = np.linalg.inv(affine)
    return np.asarray(world, dtype=float) @ inv[:3, :3].T + inv[:3, 3]


def to_world(affine: np.ndarray, vox: np.ndarray) -> np.ndarray:
    """Индексы вокселей -> мировые мм по affine."""
    vox = np.asarray(vox, dtype=float)
    return vox @ affine[:3, :3].T + affine[:3, 3]


def voxel_index(affine: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Мир -> ближайший целый индекс вокселя (X,Y,Z)."""
    return np.rint(to_voxel(affine, world)).astype(int)


# --------------------------------------------------------------------------- #
#  Чтение данных                                                               #
# --------------------------------------------------------------------------- #

def load_sample(scan_id: str, vessel: str) -> dict:
    path = sample_path(scan_id, vessel)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_ct(scan_id: str) -> tuple[np.ndarray, np.ndarray]:
    """КТ (X,Y,Z) float32 + affine (RAS)."""
    import nibabel as nib

    img = nib.load(str(ct_path(scan_id)))
    return np.asanyarray(img.dataobj).astype(np.float32), img.affine


def load_mask(path) -> tuple[np.ndarray, np.ndarray]:
    """Бинарная маска (X,Y,Z) bool + affine."""
    import nibabel as nib

    img = nib.load(str(path))
    return np.asanyarray(img.dataobj) > 0, img.affine


# --------------------------------------------------------------------------- #
#  Ресемплинг SeqSeg-маски на нативную сетку КТ                                #
# --------------------------------------------------------------------------- #

def resample_to_ct(mha_path, scan_id: str) -> np.ndarray:
    """Читает .mha и ресемплит на сетку КТ скана. Возвращает (z, y, x) bool.

    SeqSeg пишет результат в физическом пространстве исходной КТ (возможно, на
    более мелкой изотропной сетке), поэтому ресемплим NearestNeighbor к сетке КТ.
    """
    import SimpleITK as sitk

    ref = sitk.ReadImage(str(ct_path(scan_id)))
    src = sitk.ReadImage(str(mha_path))
    if src.GetSize() != ref.GetSize() or not np.allclose(
            src.GetSpacing(), ref.GetSpacing()):
        src = sitk.Resample(src, ref, sitk.Transform(),
                            sitk.sitkNearestNeighbor, 0, src.GetPixelID())
    return sitk.GetArrayFromImage(src) > 0


# --------------------------------------------------------------------------- #
#  Проекции (совместимо с pfai_gen/generator/common.py)                        #
# --------------------------------------------------------------------------- #

def grey_from_hu(ct: np.ndarray, window=HU_WINDOW,
                 gamma: float = GREY_GAMMA) -> np.ndarray:
    lo, hi = window
    x = np.clip((ct - lo) / (hi - lo), 0.0, 1.0)
    return np.power(x, gamma, dtype=np.float32)


def orient_mip(volume, drop: int, row: int, col: int, affine) -> tuple:
    """MIP вдоль оси drop, развёрнутый так, что строки=row, столбцы=col.

    Возвращает (изображение, flip_row, flip_col).
    """
    rem = [i for i in range(3) if i != drop]
    order = (rem.index(row), rem.index(col))
    proj = volume.max(axis=drop)
    if volume.ndim == 4:
        order = order + (2,)
        proj = np.transpose(proj, order)
    else:
        proj = np.transpose(proj, order)
    flip_r = bool(affine[row, row] < 0)
    flip_c = bool(affine[col, col] < 0)
    if flip_r:
        proj = proj[::-1]
    if flip_c:
        proj = proj[:, ::-1]
    return proj, flip_r, flip_c


def voxel_in_view(points, affine, row, col, shape, flip_r, flip_c):
    """Мировые точки (X,Y,Z) -> (col, row) в развёрнутом изображении."""
    vox = to_voxel(affine, points)
    c = vox[:, col].copy()
    r = vox[:, row].copy()
    if flip_r:
        r = (shape[0] - 1) - r
    if flip_c:
        c = (shape[1] - 1) - c
    return c, r


def view_limits(mask, affine, margin_mm: float = 25.0) -> dict:
    """Границы просмотра по каждой оси: bbox маски + запас, с учётом флипов."""
    idx = np.argwhere(mask > 0)
    n = np.array(mask.shape)
    lo = idx.min(axis=0) if idx.size else np.zeros(3, int)
    hi = idx.max(axis=0) if idx.size else n - 1
    spacing = np.abs(np.diag(affine))[:3]
    limits = {}
    for a in range(3):
        m = int(np.ceil(margin_mm / spacing[a])) if spacing[a] > 0 else 0
        a0, a1 = max(0, int(lo[a]) - m), min(n[a] - 1, int(hi[a]) + m)
        limits[a] = ((n[a] - 1 - a1, n[a] - 1 - a0)
                     if affine[a, a] < 0 else (a0, a1))
    return limits
