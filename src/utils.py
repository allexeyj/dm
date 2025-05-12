import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.simulation import simulate_sample
from src.graph_builders import build_knn_graph, build_distance_graph
from src.features import compute_feature


def single_run(dist, n, graph_type, param, feature_name):
    """
    Выполняет один прогон эксперимента:
    - генерирует выборку,
    - строит граф по выборке,
    - вычисляет заданную характеристику графа.

    Параметры:
    ----------
    dist : tuple
        Пара (название распределения, параметры) для генерации выборки.
    n : int
        Размер выборки.
    graph_type : str
        Тип графа: "knn" или "distance".
    param : int or float
        Параметр графа (k или порог расстояния).
    feature_name : str
        Название вычисляемой характеристики графа.

    Возвращает:
    ----------
    float
        Значение характеристики графа.
    """

    data = simulate_sample(n, *dist)
    G = (
        build_knn_graph(data, param)
        if graph_type == "knn"
        else build_distance_graph(data, param)
    )
    return compute_feature(G, feature_name)


# Функция для полного эксперимента по сетке (n × param)
def run_experiment(
    dist0,
    dist1,
    sample_sizes,
    params,
    feature_name,
    n_sim=500,
    graph_type="knn",
    alpha=0.055,
    seed=None,
):
    """
    Проводит серию экспериментов для двух распределений и возвращает метрики мощности критерия.

    Параметры:
    ----------
    dist0, dist1 : tuple
        Пара (распределение, параметры), соответствующие H0 и H1.
    sample_sizes : list[int]
        Список размеров выборки n.
    params : list[int or float]
        Список параметров графа (k или расстояние).
    feature_name : str
        Название вычисляемой характеристики графа.
    n_sim : int
        Кол-во симуляций для каждой пары (n, param).
    graph_type : str
        Тип графа: "knn" или "distance".
    alpha : float
        Уровень значимости (для порога по H0).
    seed : int, optional
        Значение для инициализации генератора случайных чисел.

    Возвращает:
    ----------
    pd.DataFrame
        Результаты эксперимента: средние и дисперсии под H0 и H1, порог и мощность.
    """

    """
    Возвращает DataFrame со столбцами:
      n, param,
      mean_H0, var_H0,
      mean_H1, var_H1,
      threshold, power
    """
    if seed is not None:
        np.random.seed(seed)

    if graph_type not in ("knn", "distance"):
        raise ValueError(f"Unknown graph_type: {graph_type}")
    records = []
    # Проходим по всем сочетаниям n и param с индикатором прогресса
    for n in tqdm(sample_sizes, desc="Sample sizes"):
        for p in tqdm(params, desc=f"Params for n={n}", leave=False):
            t0 = [
                single_run(dist0, n, graph_type, p, feature_name) for _ in range(n_sim)
            ]
            t1 = [
                single_run(dist1, n, graph_type, p, feature_name) for _ in range(n_sim)
            ]
            m0, v0 = np.mean(t0), np.var(t0)
            m1, v1 = np.mean(t1), np.var(t1)
            # Используем ceil-индекс для (1-α)-квантили
            t0_sorted = np.sort(t0)
            idx = int(np.ceil((1 - alpha) * n_sim)) - 1
            # защита от выхода за границы
            if idx < 0:
                idx = 0
            elif idx >= n_sim:
                idx = n_sim - 1
            threshold = t0_sorted[idx]

            power = np.mean(np.array(t1) > threshold)
            records.append(
                {
                    "n": n,
                    "param": p,
                    "mean_H0": m0,
                    "var_H0": v0,
                    "mean_H1": m1,
                    "var_H1": v1,
                    "threshold": threshold,
                    "power": power,
                }
            )
    return pd.DataFrame(records)
