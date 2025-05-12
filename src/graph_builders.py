from __future__ import annotations
import itertools
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


def build_knn_graph(samples: np.ndarray, k: int) -> nx.Graph:
    """
    Строит симметричный kNN-граф (в 1D) по выборке.

    Параметры:
    ----------
    samples : np.ndarray
        Одномерный массив данных.
    k : int
        Количество ближайших соседей.

    Возвращает:
    ----------
    nx.Graph
        Получившийся граф.
    """

    if not isinstance(k, int):
        raise TypeError("k must be int")

    samples = np.asarray(samples)
    n = samples.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 0:
        return G

    if k < 1:
        raise ValueError("k must be at least 1")
    if k >= n:
        raise ValueError(f"k must be less than n (got k={k}, n={n})")

    if k == n - 1:
        G.add_edges_from(itertools.combinations(range(n), 2))
        return G

    pts = samples.reshape(-1, 1)
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)

    for i in range(n):
        neigh_i = idx[i, 1:]
        for j in neigh_i:
            if i in idx[j, 1:] and i < j:
                G.add_edge(i, j)
    return G


def build_distance_graph(samples: np.ndarray, d: float) -> nx.Graph:
    """
    Строит граф по расстоянию: соединяет точки, если они ближе d.

    Параметры:
    ----------
    samples : np.ndarray
        Одномерный массив данных.
    d : float
        Порог расстояния.

    Возвращает:
    ----------
    nx.Graph
        Получившийся граф.
    """

    if not isinstance(d, (int, float)):
        raise TypeError("d must be a non-negative number")
    if d < 0:
        raise ValueError("distance threshold d must be non-negative")

    samples = np.asarray(samples)
    n = samples.shape[0]

    G = nx.Graph()
    G.add_nodes_from(range(n))
    if n == 0:
        return G

    # Для d == inf сразу полный граф
    if d == np.inf:
        G.add_edges_from(itertools.combinations(range(n), 2))
        return G

    pts = samples.reshape(-1, 1)
    tree = cKDTree(pts)
    edges = tree.query_pairs(r=d, p=2)
    G.add_edges_from(edges)
    return G
