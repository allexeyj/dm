import pytest
import networkx as nx

from src.features import compute_feature, SUPPORTED_FEATURES


# --------------------------------------------------------------------------- #
# 1. Базовые / ранее существовавшие тесты
# --------------------------------------------------------------------------- #
def test_compute_feature_on_empty():
    G = nx.Graph()
    assert compute_feature(G, "num_components") == 0
    assert compute_feature(G, "max_degree") == 0
    assert compute_feature(G, "triangle_count") == 0


@pytest.fixture
def triangle_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


@pytest.fixture
def complex_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])  # квадрат с диагональю
    return G


def test_triangle_count(triangle_graph):
    assert compute_feature(triangle_graph, "triangle_count") == 1


def test_num_components_multiple():
    G = nx.disjoint_union(nx.path_graph(2), nx.path_graph(3))
    assert compute_feature(G, "num_components") == 2


def test_all_features_on_triangle():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])  # triangle
    assert compute_feature(G, "triangle_count") == 1
    assert compute_feature(G, "max_degree") == 2
    assert compute_feature(G, "num_components") == 1


def test_errors():
    with pytest.raises(TypeError):
        compute_feature("not a graph", "triangle_count")
    with pytest.raises(ValueError):
        compute_feature(nx.path_graph(2), "unknown_feature")


def test_feature_type_errors():
    with pytest.raises(TypeError):
        compute_feature("not_a_graph", "max_degree")
    with pytest.raises(ValueError):
        compute_feature(nx.path_graph(2), "nonexistent_feature")


def test_max_independent_set_empty_and_simple():
    # Пустой граф → 0
    G0 = nx.Graph()
    assert compute_feature(G0, "max_independent_set") == 0

    # Путь на 3 вершинах: max independent set = 2 (узлы 0 и 2)
    P3 = nx.path_graph(3)
    assert compute_feature(P3, "max_independent_set") == 2


def test_max_independent_set_triangle():
    # Треугольник: любая независимая вершина одна → размер = 1
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    assert compute_feature(G, "max_independent_set") == 1


# --------------------------------------------------------------------------- #
# 2. Новые комплексные тесты
# --------------------------------------------------------------------------- #


# ---------- 2.1 Одновершинный граф ---------------------------------------- #
def test_single_vertex_graph():
    G = nx.Graph()
    G.add_node(0)
    expected = {
        "max_degree": 0.0,
        "min_degree": 0.0,
        "num_components": 1.0,
        "triangle_count": 0.0,
        "max_independent_set": 1.0,
        "chromatic_number": 1.0,
        "clique_number": 1.0,
        "domination_number": 1.0,
        "clique_cover_number": 1.0,
        "articulation_points": 0.0,
    }
    for fname, value in expected.items():
        assert compute_feature(G, fname) == value, f"Failed on {fname}"


# ---------- 2.2 Полный граф K_n ------------------------------------------ #
@pytest.mark.parametrize("n", [2, 4, 6])
def test_complete_graph_features(n):
    K = nx.complete_graph(n)
    assert compute_feature(K, "max_degree") == n - 1
    assert compute_feature(K, "min_degree") == n - 1
    assert compute_feature(K, "triangle_count") == float(n * (n - 1) * (n - 2) / 6)
    assert compute_feature(K, "clique_number") == n
    assert compute_feature(K, "chromatic_number") == n  # greedy на K_n выдаёт n
    assert compute_feature(K, "clique_cover_number") == 1  # единая клика покрывает всё
    assert compute_feature(K, "domination_number") == 1  # любая вершина доминирует всё


# ---------- 2.3 Пустой граф (n вершин, 0 рёбер) --------------------------- #
@pytest.mark.parametrize("n", [3, 7])
def test_empty_graph_features(n):
    G = nx.empty_graph(n)
    assert compute_feature(G, "max_degree") == 0
    assert compute_feature(G, "min_degree") == 0
    assert (
        compute_feature(G, "num_components") == n
    )  # каждая вершина отдельный компонент
    assert compute_feature(G, "clique_number") == 1 if n else 0
    assert compute_feature(G, "chromatic_number") == 1  # все вершины одного цвета
    assert compute_feature(G, "clique_cover_number") == n  # нужно n клик-из-вершин
    assert compute_feature(G, "triangle_count") == 0


# ---------- 2.4 Звезда ---------------------------------------------------- #
@pytest.mark.parametrize("leaves", [1, 5, 10])
def test_star_graph(leaves):
    S = nx.star_graph(leaves)  # центр = 0, листья = 1..leaves
    assert compute_feature(S, "max_degree") == leaves
    assert compute_feature(S, "min_degree") == 1 if leaves else 0
    assert compute_feature(S, "domination_number") == 1  # центр доминирует
    assert compute_feature(S, "articulation_points") == (1 if leaves > 1 else 0)
    assert compute_feature(S, "clique_number") == 2 if leaves > 0 else 1
    # в звезде нет треугольников
    assert compute_feature(S, "triangle_count") == 0


# ---------- 2.5 Путь и цикл ---------------------------------------------- #
def test_path_and_cycle():
    P4 = nx.path_graph(4)  # 0-1-2-3
    C4 = nx.cycle_graph(4)  # цикл длиной 4
    # Путь
    assert compute_feature(P4, "chromatic_number") == 2
    assert compute_feature(P4, "articulation_points") == 2  # вершины 1 и 2
    # Цикл
    assert compute_feature(C4, "chromatic_number") == 2  # 2-раскрашиваемый
    assert compute_feature(C4, "articulation_points") == 0  # цикл 2-связен


# ---------- 2.6 Клик-ковер / комплемент ---------------------------------- #
def test_clique_cover_vs_chromatic_of_complement():
    G = nx.erdos_renyi_graph(8, 0.3, seed=42)
    clique_cover = compute_feature(G, "clique_cover_number")
    # Проверяем: clique_cover_number(G) == χₐ(Ĝ)
    chrom_comp = compute_feature(nx.complement(G), "chromatic_number")
    assert clique_cover == chrom_comp


# ---------- 2.7 Точки сочленения (articulation points) ------------------- #
def test_articulation_points_custom():
    G = nx.Graph()
    G.add_edges_from(
        [
            (0, 1),
            (1, 2),  # цепочка
            (1, 3),
            (3, 4),
            (4, 5),  # другая цепочка, ключевое ребро 1-3
        ]
    )
    # Точки сочленения: 1, 3, 4
    assert compute_feature(G, "articulation_points") == 3


# ---------- 2.8 Все признаки на случайном графе не выбрасывают ошибки ---- #
@pytest.mark.parametrize("p", [0.0, 0.1, 0.5, 1.0])
def test_random_graph_all_features(p):
    G = nx.erdos_renyi_graph(12, p, seed=123)
    for fname in SUPPORTED_FEATURES:
        # главное — чтобы функция не бросала исключений и возвращала float
        val = compute_feature(G, fname)
        assert isinstance(val, float), f"{fname} did not return float"


# ---------- 2.9 Идиопатические случаи ошибок ----------------------------- #
def test_negative_behaviour_unknown_feature():
    G = nx.path_graph(2)
    with pytest.raises(ValueError):
        compute_feature(G, "this_feature_does_not_exist")


def test_non_graph_input_all_supported_names():
    for fname in SUPPORTED_FEATURES:
        with pytest.raises(TypeError):
            compute_feature(123, fname)
