# HNSW

Реализация NSW и HNSW (Hierarchical Navigable Small World) для приближённого поиска ближайших соседей на `numpy`.

## Структура

```
.
├── graph.py        # общие части: search_layer, select_simple, select_heuristic
├── nsw.py          # NSW — однослойный граф
├── hnsw.py         # HNSW — многослойный граф
├── metrics.py      # метрики расстояния (euclidean, manhattan)
├── bruteforce.py   # точный kNN перебором, используется как эталон
├── data.py         # генерация данных (uniform, clusters)
├── bench.py        # подсчёт recall@k, числа вычислений расстояния, sweep по параметрам
├── plots.py        # отрисовка графов, поиска и графиков recall/стоимость
├── figures.py      # сборка всех картинок в папку figures/
└── tests/          # тесты на pytest
```

## Запуск

Зависимости ставятся через [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Пример использования:

```python
import numpy as np
from hnsw import HNSW

points = np.random.default_rng(0).uniform(size=(1000, 2))

index = HNSW(points, m=8, ef_construction=40, seed=0)
index.build()

index.search(np.array([0.5, 0.5]), k=10, ef=50)  # [(distance, index), ...]
```

Построить картинки (сохраняются в `figures/`):

```bash
uv run python figures.py
```

Запустить тесты:

```bash
uv run pytest
```

## AI Usage

AI использовался только для вспомогательных вещей, которые не влияют на работу алгоритма: графики и визуализация (`plots.py`, `figures.py`), тесты (`tests/`) и бенчмарк-обвязка, а также часть этого ридми :). Также в некоторых местах использовался AI для написания докстрингов.

Сами NSW и HNSW (`graph.py`, `nsw.py`, `hnsw.py`, `metrics.py`) написаны вручную.
