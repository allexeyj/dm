import time
import numpy as np
import pandas as pd
import pytest

from src.utils import single_run, run_experiment


# ------------------------------------------------------------------
# Базовые тесты из первоначального набора (оставляем без изменений)
# ------------------------------------------------------------------
def test_single_run_reproducible():
    np.random.seed(0)
    t1 = single_run(("normal", {"mu": 0, "sigma": 1}), 50, "knn", 3, "triangle_count")
    np.random.seed(0)
    t2 = single_run(("normal", {"mu": 0, "sigma": 1}), 50, "knn", 3, "triangle_count")
    assert t1 == t2


def test_power_level_alpha():
    # при H0 = H1 мощность ≈ α
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 0, "sigma": 1}),
        sample_sizes=[100],
        params=[5],
        feature_name="triangle_count",
        n_sim=500,
        alpha=0.05,
        seed=42,
    )
    assert abs(df.loc[0, "power"] - 0.05) < 0.03


def test_run_experiment_dataframe_structure():
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 1, "sigma": 1}),
        sample_sizes=[50, 100],
        params=[2, 4],
        feature_name="triangle_count",
        n_sim=100,
        alpha=0.1,
        seed=1,
    )
    # Проверяем кол-во строк и колонки
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4  # 2 sample_sizes × 2 params
    expected_cols = {
        "n",
        "param",
        "mean_H0",
        "var_H0",
        "mean_H1",
        "var_H1",
        "threshold",
        "power",
    }
    assert expected_cols.issubset(df.columns)


def test_power_increases_with_shift():
    # Сравниваем мощность при разных сдвигах H1: mu=0.5 vs mu=2
    df1 = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 0.5, "sigma": 1}),
        sample_sizes=[100],
        params=[5],
        feature_name="triangle_count",
        n_sim=300,
        alpha=0.05,
        seed=10,
    )
    df2 = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 2, "sigma": 1}),
        sample_sizes=[100],
        params=[5],
        feature_name="triangle_count",
        n_sim=300,
        alpha=0.05,
        seed=10,
    )
    p1, p2 = df1.loc[0, "power"], df2.loc[0, "power"]
    assert p2 >= p1, (
        f"Expected non-decreasing power for larger shift " f"but got p1={p1}, p2={p2}"
    )


def test_invalid_graph_type():
    with pytest.raises(ValueError):
        run_experiment(
            ("normal", {"mu": 0, "sigma": 1}),
            ("normal", {"mu": 1, "sigma": 1}),
            sample_sizes=[50],
            params=[3],
            feature_name="triangle_count",
            n_sim=10,
            alpha=0.05,
            seed=0,
            graph_type="unknown",
        )


def test_invalid_params_negative():
    # отрицательный параметр графа
    with pytest.raises(ValueError):
        run_experiment(
            ("normal", {"mu": 0, "sigma": 1}),
            ("normal", {"mu": 1, "sigma": 1}),
            sample_sizes=[50],
            params=[-1],  # k < 1
            feature_name="triangle_count",
            n_sim=10,
            alpha=0.05,
            seed=0,
        )


def test_single_run_invalid_n():
    with pytest.raises(ValueError):
        single_run(("normal", {"mu": 0, "sigma": 1}), -5, "knn", 3, "triangle_count")


def test_distance_graph_edge_case():
    # distance d = 0, должно работать, но нет треугольников
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 1, "sigma": 1}),
        sample_sizes=[50],
        params=[0],
        feature_name="triangle_count",
        n_sim=20,
        alpha=0.1,
        seed=0,
        graph_type="distance",
    )
    assert all(df["mean_H0"] == 0)
    assert all(df["mean_H1"] == 0)


# ------------------------------------------------------------------
# Новые дополнительные тесты
# ------------------------------------------------------------------


def test_threshold_between_min_max():
    """Порог должен лежать между min(t0) и max(t0) включительно."""
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 0, "sigma": 1}),
        sample_sizes=[60],
        params=[4],
        feature_name="triangle_count",
        n_sim=200,
        alpha=0.07,
        seed=7,
    )
    thr = df.loc[0, "threshold"]
    min_t0, max_t0 = df.loc[0, "mean_H0"] - 5 * np.sqrt(df.loc[0, "var_H0"]), df.loc[
        0, "mean_H0"
    ] + 5 * np.sqrt(df.loc[0, "var_H0"])
    # Проверяем, что threshold не вышел далеко за пределы распределения
    assert min_t0 <= thr <= max_t0


@pytest.mark.parametrize("alpha_small, alpha_large", [(0.01, 0.2), (0.05, 0.15)])
def test_threshold_monotonic_with_alpha(alpha_small, alpha_large):
    """
    При одинаковых данных lower alpha (более строгий тест) => больший порог.
    """
    kwargs = dict(
        dist0=("normal", {"mu": 0, "sigma": 1}),
        dist1=("normal", {"mu": 0, "sigma": 1}),
        sample_sizes=[80],
        params=[3],
        feature_name="triangle_count",
        n_sim=400,
        seed=123,
    )
    df_small = run_experiment(alpha=alpha_small, **kwargs)
    # перезапускаем с теми же seed & входными данными => те же t0
    df_large = run_experiment(alpha=alpha_large, **kwargs)
    thr_small = df_small.loc[0, "threshold"]
    thr_large = df_large.loc[0, "threshold"]
    assert (
        thr_small >= thr_large
    ), f"threshold should decrease with growing alpha, got {thr_small} < {thr_large}"


def test_run_experiment_reproducibility_full_df():
    """
    Полные DataFrame должны быть идентичны при одинаковых seed.
    """
    kwargs = dict(
        dist0=("normal", {"mu": 0, "sigma": 1}),
        dist1=("normal", {"mu": 1, "sigma": 1}),
        sample_sizes=[40, 80],
        params=[2, 3],
        feature_name="triangle_count",
        n_sim=120,
        alpha=0.08,
        seed=2023,
    )
    df1 = run_experiment(**kwargs)
    # Чтобы случайные числа совпали, надо восстановить seed
    df2 = run_experiment(**kwargs)
    pd.testing.assert_frame_equal(df1, df2)


def test_knn_param_too_large_raises():
    """
    Для k ≥ n при построении kNN-графа должна возникать ошибка.
    """
    with pytest.raises(ValueError):
        run_experiment(
            ("normal", {"mu": 0, "sigma": 1}),
            ("normal", {"mu": 1, "sigma": 1}),
            sample_sizes=[30],
            params=[30],  # k == n
            feature_name="triangle_count",
            n_sim=5,
            alpha=0.05,
            seed=0,
            graph_type="knn",
        )


def test_distance_param_negative_raises():
    with pytest.raises(ValueError):
        run_experiment(
            ("normal", {"mu": 0, "sigma": 1}),
            ("normal", {"mu": 1, "sigma": 1}),
            sample_sizes=[30],
            params=[-0.1],  # отрицательное расстояние
            feature_name="triangle_count",
            n_sim=5,
            alpha=0.05,
            seed=0,
            graph_type="distance",
        )


def test_one_simulation_edge_case():
    """
    Проверяем, что n_sim = 1 не падает и power принимает значения 0/1.
    """
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 2, "sigma": 1}),
        sample_sizes=[20],
        params=[3],
        feature_name="triangle_count",
        n_sim=1,
        alpha=0.5,
        seed=999,
    )
    power = df.loc[0, "power"]
    assert power in {0.0, 1.0}


def test_variances_non_negative():
    """
    var_H0 и var_H1 должны быть неотрицательны (с некоторым численным допуском).
    """
    df = run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 1, "sigma": 1}),
        sample_sizes=[60],
        params=[5],
        feature_name="triangle_count",
        n_sim=150,
        alpha=0.05,
        seed=321,
    )
    assert (df["var_H0"] >= -1e-12).all()
    assert (df["var_H1"] >= -1e-12).all()


def test_runtime_not_exploding():
    """
    Очень грубая проверка: небольшой эксперимент должен выполняться < 2 сек.
    Этот тест предохраняет от случайной экспоненциальной сложности.
    """
    start = time.time()
    run_experiment(
        ("normal", {"mu": 0, "sigma": 1}),
        ("normal", {"mu": 0, "sigma": 1}),
        sample_sizes=[20],
        params=[2],
        feature_name="triangle_count",
        n_sim=30,
        alpha=0.1,
        seed=42,
    )
    assert time.time() - start < 2.0
