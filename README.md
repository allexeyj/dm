# Graph Analysis Toolkit

[![Python CI](https://github.com/allexeyj/dm/actions/workflows/python-ci.yml/badge.svg)](https://github.com/allexeyj/dm/actions/workflows/python-ci.yml)

Набор утилит для генерации случайных выборок, построения графов (k-NN и ε-графов) и вычисления их признаков. Подходит для анализа различий между распределениями с помощью структурных характеристик графов.

---

## 📁 Структура проекта

```

/src
├── simulation.py        # генерация выборок из разных распределений
├── graph_builders.py    # построение k-NN и ε-графов
├── features.py          # вычисление признаков (χ, α, τ и др.)
├── utils.py             # функция run_experiment
└── __init__.py

/tests
├── test_simulation.py
├── test_graph_builders.py
├── test_features.py
└── test_experiment.py

/notebooks
├── part_1_exploration.ipynb  # Часть I (Alex): Laplace vs Normal

/report-alex
├── csv-files/
├── pics/
└── …


README.md           
requirements.txt

````

---

## ✅ Установка зависимостей

```bash
pip install -r requirements.txt
````

---

## 🧪 Запуск тестов

```bash
pytest
```

---

## 📓 Запуск ноутбуков

```bash
jupyter notebook notebooks/part_1_exploration.ipynb
```

---

## 🧠 Пример использования

```python
from src.simulation import simulate_sample
from src.graph_builders import build_knn_graph, build_distance_graph
from src.features import compute_feature

# 1) Сгенерировать 100 точек из N(0,1)
data = simulate_sample(100, 'normal', {'mu': 0, 'sigma': 1})

# 2) Построить k-NN граф (k=5)
G_knn = build_knn_graph(data, k=5)
print("Число треугольников в k-NN графе:", compute_feature(G_knn, "triangle_count"))

# 3) Построить ε-граф с порогом d=0.5
G_dist = build_distance_graph(data, d=0.5)
print("Максимальная степень в ε-графе:", compute_feature(G_dist, "max_degree"))
```

---

## ℹ️ Авторы

Проект командный. Часть I выполнена отдельно двумя участниками:

* **Alexey Shaturnyy**: Laplace vs Normal
* **Dmitriy Kutcenko**: Pareto vs Exponential

Общие модули, тесты и документация разрабатывались совместно.
