import math
import numpy as np
import pytest

from src.simulation import simulate_sample


# ------------- базовые проверки из исходного файла -----------------
def test_simulate_sample_zero_length():
    arr = simulate_sample(0, "normal", {"mu": 0, "sigma": 1})
    assert isinstance(arr, np.ndarray)
    assert arr.size == 0


def test_simulate_sample_normal_stats():
    np.random.seed(42)
    arr = simulate_sample(10_000, "normal", {"mu": 5, "sigma": 2})
    mean = np.mean(arr)
    var = np.var(arr)
    assert abs(mean - 5) < 0.1
    assert abs(var - 4) < 0.2


def test_simulate_sample_negative_params():
    with pytest.raises(ValueError):
        simulate_sample(-1, "normal", {"mu": 0, "sigma": 1})
    with pytest.raises(ValueError):
        simulate_sample(5, "unknown", {})


def test_simulate_sample_laplace_stats():
    beta0 = math.sqrt(0.5)  # Var = 2*beta^2 = 1
    np.random.seed(123)
    arr = simulate_sample(100_000, "laplace", {"mu": 0, "beta": beta0})
    mean, var = np.mean(arr), np.var(arr)
    assert abs(mean) < 0.02
    assert abs(var - 1) < 0.05


def test_simulate_sample_pareto_default_alpha():
    n = 1_000
    samp = simulate_sample(n, "pareto", {})
    assert isinstance(samp, np.ndarray)
    assert samp.shape == (n,)
    assert np.all(samp >= 1.0)


def test_simulate_sample_pareto_custom_alpha():
    n = 5_000
    samp_low = simulate_sample(n, "pareto", {"alpha": 1.5})
    samp_high = simulate_sample(n, "pareto", {"alpha": 5.0})
    assert samp_low.mean() > samp_high.mean()


def test_simulate_sample_exponential_default_lambda():
    n = 1_000
    samp = simulate_sample(n, "exponential", {})
    assert isinstance(samp, np.ndarray)
    assert samp.shape == (n,)
    assert np.all(samp >= 0.0)


def test_simulate_sample_exponential_custom_lambda():
    n = 5_000
    slow = simulate_sample(n, "exponential", {"lam": 1.0})
    fast = simulate_sample(n, "exponential", {"lam": 5.0})
    assert slow.mean() > fast.mean()


def test_simulate_sample_zero_length_pareto_and_exponential():
    assert simulate_sample(0, "pareto", {}).size == 0
    assert simulate_sample(0, "exponential", {}).size == 0


def test_simulate_sample_invalid_dist_raises():
    with pytest.raises(ValueError):
        simulate_sample(10, "pareto", {"alpha": -1.0})
    with pytest.raises(ValueError):
        simulate_sample(10, "exponential", {"lam": -2.0})


# ------------- новые, дополнительные тесты -------------------------


@pytest.mark.parametrize(
    "dist, params",
    [
        ("normal", {"mu": 0, "sigma": 1}),
        ("laplace", {"mu": 0, "beta": 1}),
        ("pareto", {"alpha": 3}),
        ("exponential", {"lam": 2}),
    ],
)
def test_output_type_and_shape(dist, params):
    """Тип numpy.ndarray, одномерность, dtype=float64."""
    n = 137
    x = simulate_sample(n, dist, params)
    assert isinstance(x, np.ndarray)
    assert x.shape == (n,)
    assert np.issubdtype(x.dtype, np.floating)


@pytest.mark.parametrize("n", [1, 2, 10])
@pytest.mark.parametrize(
    "dist, params",
    [
        ("normal", {"mu": -10, "sigma": 0.1}),
        ("laplace", {"mu": 5, "beta": 0.5}),
        ("exponential", {"lam": 0.3}),
        ("pareto", {"alpha": 10}),
    ],
)
def test_small_n_no_crash(n, dist, params):
    """n=1,2,… должно работать без ошибок и возвращать правильный размер."""
    out = simulate_sample(n, dist, params)
    assert out.size == n


def test_reproducibility_with_same_seed():
    np.random.seed(777)
    a = simulate_sample(100, "normal", {"mu": 3.3, "sigma": 0.77})
    np.random.seed(777)
    b = simulate_sample(100, "normal", {"mu": 3.3, "sigma": 0.77})
    assert np.array_equal(a, b)


def test_normal_variance_scale_relation():
    """Удвоение σ должно ~в 4 раза увеличивать дисперсию."""
    n = 50_000
    np.random.seed(1)
    s1 = simulate_sample(n, "normal", {"mu": 0, "sigma": 1})
    s2 = simulate_sample(n, "normal", {"mu": 0, "sigma": 2})
    ratio = np.var(s2) / np.var(s1)
    assert 3.6 < ratio < 4.4  # ≈4 с небольшими отклонениями


def test_laplace_mean_and_variance_default():
    """Laplace(0,1): E=0, Var=2."""
    np.random.seed(2024)
    x = simulate_sample(200_000, "laplace", {})
    mean, var = x.mean(), x.var()
    assert abs(mean) < 0.02
    assert abs(var - 2) < 0.05


def test_pareto_default_theoretical_mean():
    """Для alpha=3 и scale=1: E = α/(α-1) = 1.5."""
    np.random.seed(321)
    n = 500_000
    x = simulate_sample(n, "pareto", {})
    assert abs(x.mean() - 1.5) < 0.03


def test_exponential_default_theoretical_mean_var():
    lam = 2 / math.sqrt(3)
    true_mean = 1 / lam
    true_var = 1 / lam**2
    np.random.seed(999)
    x = simulate_sample(300_000, "exponential", {})
    assert abs(x.mean() - true_mean) < 0.02
    assert abs(x.var() - true_var) < 0.05


def test_invalid_parameter_types():
    with pytest.raises(TypeError):
        simulate_sample(10.5, "normal", {"mu": 0, "sigma": 1})  # n не int
    with pytest.raises(TypeError):
        simulate_sample(5, "normal", {"mu": "zero", "sigma": 1})


def test_large_scale_parameters():
    """Очень большой scale/σ/β не должны приводить к ошибкам."""
    for dist, param_name in [("normal", "sigma"), ("laplace", "beta")]:
        arr = simulate_sample(1000, dist, {param_name: 1e6})
        assert np.isfinite(arr).all()


def test_vectorization_many_calls_quick():
    """Просто проверяем, что 100 одновременных вызовов не выбивают память."""
    res = [simulate_sample(1000, "normal", {"mu": i, "sigma": 1}) for i in range(100)]
    assert len(res) == 100
    # лёгкая sanity-check: среднее каждой выборки ≈ своему μ
    ok = [abs(r.mean() - i) < 0.1 for i, r in enumerate(res)]
    assert all(ok)
