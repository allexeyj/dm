import numpy as np


def simulate_sample(n: int, dist: str, params: dict) -> np.ndarray:
    """
    Генерирует выборку заданного распределения.

    Параметры:
    ----------
    n : int
        Размер выборки.
    dist : str
        Название распределения: "normal", "laplace", "pareto", "exponential".
    params : dict
        Параметры распределения:
        - normal: {"mu", "sigma"}
        - laplace: {"mu", "beta"}
        - pareto: {"alpha"}
        - exponential: {"lam"}

    Возвращает:
    ----------
    np.ndarray
        Сгенерированная выборка.
    """

    if not isinstance(n, int):
        raise TypeError("Sample size n must be int")
    if n < 0:
        raise ValueError("Sample size n must be non-negative")
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    for key, val in params.items():
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"Parameter {key!r} must be a number, got {type(val).__name__}"
            )

    if dist == "normal":
        return np.random.normal(
            loc=params.get("mu", 0), scale=params.get("sigma", 1), size=n
        )
    elif dist == "laplace":
        return np.random.laplace(
            loc=params.get("mu", 0), scale=params.get("beta", 1), size=n
        )
    elif dist == "pareto":
        return np.random.pareto(a=params.get("alpha", 3.0), size=n) + 1
    elif dist == "exponential":
        return np.random.exponential(
            scale=1 / params.get("lam", 2.0 / np.sqrt(3)), size=n
        )
    else:
        raise ValueError(f"Unknown distribution: {dist}")
