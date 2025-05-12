import itertools

import numpy as np
import networkx as nx
import pytest

from src.graph_builders import build_knn_graph, build_distance_graph


def _naive_mutual_knn(samples: np.ndarray, k: int) -> set[tuple[int, int]]:
    """
    «Честный» (O(n² log n)) вариант построения взаимного k-NN-графа
    для проверки корректности быстрого алгоритма.
    Возвращает множество неориентированных рёбер (i, j), i < j.
    """
    n = len(samples)
    dists = np.abs(samples[:, None] - samples[None, :])

    edges: set[tuple[int, int]] = set()
    for i in range(n):
        knn_i = np.argsort(dists[i])[1: k + 1]  # без самого i
        for j in knn_i:
            knn_j = np.argsort(dists[j])[1: k + 1]
            if i in knn_j and i < j:
                edges.add((i, j))
    return edges


def _naive_eps_graph(samples: np.ndarray, eps: float) -> set[tuple[int, int]]:
    n = len(samples)
    edges = set()
    for i, j in itertools.combinations(range(n), 2):
        if abs(samples[i] - samples[j]) <= eps:
            edges.add((i, j))
    return edges


def test_build_knn_graph_empty():
    G = build_knn_graph(np.array([]), 3)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 0


def test_build_knn_graph_symmetry_and_connectivity():
    data = np.array([0.0, 0.1, 0.2, 1.0])
    G = build_knn_graph(data, 2)
    assert G.has_edge(0, 1)
    assert G.has_edge(1, 2)
    assert G.has_edge(0, 2)


def test_build_knn_graph_full_on_ties():
    data = np.zeros(4)
    G = build_knn_graph(data, 3)
    n = len(data)
    assert G.number_of_edges() == n * (n - 1) // 2


def test_build_distance_graph_threshold_zero():
    data = np.array([5, 5, 5])
    G = build_distance_graph(data, 0)
    # 3 одинаковые точки → полный граф на 3 узлах
    assert G.number_of_edges() == 3


def test_build_distance_graph_various():
    data = np.array([0, 2, 4, 7, 10])
    G = build_distance_graph(data, 3)
    expected = {(0, 1), (1, 2), (2, 3)}
    assert expected.issubset(set(G.edges()))
    assert (0, 2) not in G.edges()


def test_build_graphs_invalid_params():
    with pytest.raises(ValueError):
        build_knn_graph(np.array([1, 2, 3]), 0)
    with pytest.raises(ValueError):
        build_distance_graph(np.array([0, 1]), -1)


# --------------------------------------------------------------------------- #
#                       ——— расширенный набор тестов ———                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k,n", [(1, 5), (2, 10), (4, 8)])
def test_knn_graph_matches_naive_random(k: int, n: int, seed: int = 42):
    """
    Сравниваем быстрый build_knn_graph с «честным» O(n²) решением
    на случайных данных (несколько раз, разные k и n).
    """
    rng = np.random.default_rng(seed + k + n)
    samples = rng.random(n)
    G_fast = build_knn_graph(samples, k)
    edges_fast = set(tuple(sorted(e)) for e in G_fast.edges())

    edges_slow = _naive_mutual_knn(samples, k)
    assert edges_fast == edges_slow


def test_knn_graph_k_ge_n_minus_1_full():
    data = np.linspace(0, 1, 6)
    G = build_knn_graph(data, k=len(data) - 1)
    v = len(data)
    assert G.number_of_edges() == v * (v - 1) // 2


def test_knn_graph_no_self_loops():
    data = np.linspace(0, 1, 20)
    G = build_knn_graph(data, 3)
    assert all(i != j for i, j in G.edges())


@pytest.mark.parametrize("eps", [0.5, 1.0])
def test_distance_graph_matches_naive_random(eps: float, seed: int = 123):
    n = 12
    rng = np.random.default_rng(seed + int(eps * 10))
    samples = rng.uniform(-5, 5, n)

    G_fast = build_distance_graph(samples, eps)
    edges_fast = set(tuple(sorted(e)) for e in G_fast.edges())
    edges_slow = _naive_eps_graph(samples, eps)
    assert edges_fast == edges_slow


def test_distance_graph_empty_input():
    G = build_distance_graph(np.array([]), 1.0)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_distance_graph_inf_makes_complete():
    data = np.random.default_rng(0).normal(size=7)
    G = build_distance_graph(data, np.inf)
    v = len(data)
    assert G.number_of_edges() == v * (v - 1) // 2


def test_distance_graph_no_edges_if_too_small():
    data = np.array([0.0, 100.0, 200.0])
    G = build_distance_graph(data, 0.5)
    assert G.number_of_edges() == 0


def test_graphs_are_undirected():
    data = np.linspace(0, 1, 10)
    G1 = build_knn_graph(data, 3)
    G2 = build_distance_graph(data, 0.3)

    assert G1.is_directed() is False
    assert G2.is_directed() is False


def test_randomised_stress_small():
    """
    Много мелких случайных прогонов, чтобы поймать редкие баги на ничьих.
    """
    rng = np.random.default_rng(2023)
    for _ in range(50):
        n = rng.integers(3, 15)
        k = rng.integers(1, n)
        samples = rng.normal(size=n)

        G = build_knn_graph(samples, int(k))
        assert G.number_of_nodes() == n
        # Рёбра должны быть не больше, чем C(n, 2)
        assert G.number_of_edges() <= n * (n - 1) // 2
