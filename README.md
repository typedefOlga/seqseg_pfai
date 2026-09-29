# seqseg_pfai — SeqSeg с двух концов на фрагментах pfai_gen

Пилот: для фрагмента сосуда из `pfai_gen/generator` (скан + 2 клика → маска)
запускается **SeqSeg** с обоих концов (промпт-точка старта и конца), результаты
объединяются (пересечение/объединение) и сравниваются с истинной маской
фрагмента по набору метрик. Отдельно строятся проекции в трёх анатомических
плоскостях.

Текущий кейс: **скан 908, LCx** (длина 92.1 мм, r_start 1.56 мм, r_end 0.88 мм).

## Данные (только чтение)

| Что | Где |
|---|---|
| Примеры | `/home/y.pchelintsev/vessel-seg/pfai_gen/generator/out/generate/samples/<scan>_<vessel>.json` |
| Маски фрагментов | `…/out/generate/fragments/<scan>_<vessel>.nii.gz` |
| КТ | `/srv/fast1/y.pchelitsev/datasets/ImageCAS/data/<scan>.img.nii.gz` |
| Веса SeqSeg | `/home/o.tonisheva/oc_seqseg/weights/nnUNet_results_coronary` |
| Окружение | `/home/o.tonisheva/anaconda3/envs/seqseg/bin/python` |

Координаты в JSON примеров — **RAS** (NIfTI). SeqSeg принимает сиды в **LPS**;
перевод — `ras_to_lps(p) = [-x, -y, z]` (см. `common.py`).

## Пайплайн

```bash
PY=/home/o.tonisheva/anaconda3/envs/seqseg/bin/python

# 1) сиды для двух направлений (f — от старта, b — от конца)
$PY make_seeds.py --scan 908 --vessel lcx
#    тонкий дистальный конец: --min-start-radius 1.2 — сдвинуть сид назад
#    в место с радиусом >= порога

# 2) SeqSeg (запускать только на ПОЛНОСТЬЮ свободной GPU)
nvidia-smi --query-gpu=index,memory.used --format=csv
$PY run_seqseg.py --scan 908 --vessel lcx --direction both --gpu 0
#    --direction f|b|both; --max-n-steps / --max-n-steps-per-branch / --max-n-branches

# 3) метрики против истинной маски фрагмента + варианты F, B, F∩B, F∪B
$PY evaluate.py --scan 908 --vessel lcx

# 4) проекции (сагиттальная/корональная/аксиальная)
$PY visualize.py --scan 908 --vessel lcx
```

## Файлы

| Файл | Роль |
|---|---|
| `common.py` | пути, RAS↔LPS, чтение, ресемплинг `.mha` на сетку КТ, проекции |
| `make_seeds.py` | клики → сиды SeqSeg `[old, new, r]` (два направления) |
| `run_seqseg.py` | staging (симлинки КТ + `seeds.json`) и `seqseg run batch` |
| `metrics.py` | метрики маски/топологии/центрлинии/радиуса |
| `evaluate.py` | ресемплинг, варианты, JSON/CSV |
| `visualize.py` | 3 проекции: GT vs SeqSeg |

Выход: `out/seeds/`, `out/seqseg/`, `out/eval/`, `out/viz/`.

## Метрики (`metrics.py`)

- Маска: `dice`, `precision/recall/f1`, `cl_dice`.
- Поверхности (мм): `hd`, `hd95`, `assd`, `chamfer`, `surface_dice@tol`,
  `boundary_f1/precision/recall@0.12 мм`.
- Топология (**b012**): числа Бетти `b0` (26-связные компоненты),
  `b1` (петли), `b2` (замкнутые полости) и их ошибки (`b*_err`) относительно GT.
  `b1 = b0 + b2 − χ`, где χ — число Эйлера (`skimage.measure.euler_number`).
- Центрлиния: `centerline_md`, `centerline_hd95`, `coverage@1мм`.
- Радиус: `radius_mae/rmse/bias` (EDT предсказания в точках GT-центрлинии).

## Особенности / предостережения

- SeqSeg — сид-based трекер: он идёт от сида в заданную сторону и не знает про
  границы фрагмента, поэтому может уйти в соседние ветви (большие HD/Dice
  наказывают за лишнее). Ограничивайте `--max-n-steps` / `--max-n-branches`.
- Тонкий дистальный конец (< ~1 мм) часто даёт `Labels at seed indices: []` —
  сид переносится назад в более широкое место (`--min-start-radius`).
- Выход SeqSeg `.mha` ресемплится на нативную сетку КТ (NearestNeighbor).
- Сборка в batch идёт последовательно; при многих сканах шардировать по
  `-start/-stop` на разные GPU (см. `run_seqseg.py`).
- Крупные сканы (spacing ~0.4 мм) при `CENT_MAX_SPACING=0.3` превышают лимит
  200M вокселей при финальной сборке — используйте конфиг с бо́льшим спейсингом
  (`--config-name global_coro_cent0.4`).
- Параллельные запуски разных кейсов безопасны: staging уникален
  (`staging_<scan>_<vessel>`), выходы в общем `out/seqseg/run` названы по кейсу.

## Результаты

Три кейса (LCx 908, RCA 593, LAD 306), сводка: `out/eval/summary_all.csv`,
отчёт: `out/eval/report_three_cases.md`, проекции `out/viz/<case>_masks.png`.

## Батч на 40 сканах × 3 артерии (120 фрагментов)

Считается только `UNION = F ∪ B`, метрики — на кропе `bbox(union ∪ gt)`.
Промежуточные `.mha` удаляются. Запуск на 2 GPU через tmux (сессия `seqseg`).

```bash
# запуск (пример; GPU — первые две свободные)
python run_batch.py --cases cases_120.txt --shard 0/2 --gpu 0 --union-only --crop --delete-mha
python run_batch.py --cases cases_120.txt --shard 1/2 --gpu 1 --union-only --crop --delete-mha
```

### Отслеживание прогресса (в любой момент, независимо от клиента)

- `tmux attach -t seqseg` — живые окна шардов (выход: `Ctrl-b d`; **не** убивать tmux).
- `python progress.py` — готово/всего, живые шаги f/b, ETA, ошибки, текущее среднее.
- `python aggregate.py` — средние (overall и по LAD/LCx/RCA) по уже готовым.

Файлы:
- `out/reports/metrics_union.csv` — по строке на фрагмент сразу после обработки (запись под `flock`); из него `aggregate.py` считает среднее в любой момент.
- `out/reports/metrics_mean.csv` — итоговые средние.
- `out/reports/status_*.json`, `out/reports/runtime_*.csv`, `out/reports/batch_*.log` — статус/тайминги/логи.

Драйвер **resumable**: повторный запуск пропускает готовые кейсы (с метриками)
и дорабатывает отсутствующие; упавшие кейсы повторяются отдельным проходом
(`--run-id cleanup`, свой `--gpu`).
