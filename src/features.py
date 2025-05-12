import networkx as nx
from networkx.algorithms.approximation.clique import maximum_independent_set
from networkx.algorithms.dominating import dominating_set as nx_dominating_set
from networkx.algorithms.clique import find_cliques
from networkx.algorithms.coloring import greedy_color
from networkx.algorithms.components import articulation_points


SUPPORTED_FEATURES = (
    "max_degree",
    "min_degree",
    "num_components",
    "triangle_count",
    "max_independent_set",
    "chromatic_number",
    "clique_number",
    "domination_number",
    "clique_cover_number",
    "articulation_points",
)


def compute_feature(G: nx.Graph, feature_name: str) -> float:
    """
    Вычисляет структурную характеристику неориентированного графа G.

    Поддерживаемые характеристики:
    - max_degree, min_degree, num_components, triangle_count,
    - max_independent_set, chromatic_number, clique_number,
    - domination_number, clique_cover_number, articulation_points

    Параметры:
    ----------
    G : networkx.Graph
        Неориентированный граф.
    feature_name : str
        Название характеристики.

    Возвращает:
    ----------
    float
        Значение характеристики графа.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be a networkx.Graph")

    if feature_name not in SUPPORTED_FEATURES:
        raise ValueError(
            f"Unknown feature: {feature_name!r}. Available: {SUPPORTED_FEATURES}"
        )

    # simple structural features
    if feature_name == "max_degree":
        return float(max(dict(G.degree()).values(), default=0))

    if feature_name == "min_degree":
        return float(min(dict(G.degree()).values(), default=0))

    if feature_name == "num_components":
        return float(
            nx.number_connected_components(G)
        )  # :contentReference[oaicite:5]{index=5}

    if feature_name == "triangle_count":
        # each triangle is counted at each of its three vertices
        tris = nx.triangles(G)  # :contentReference[oaicite:6]{index=6}
        return float(sum(tris.values()) / 3)

    # heavier features
    if feature_name == "max_independent_set":
        mis = maximum_independent_set(G)
        return float(len(mis))

    if feature_name == "chromatic_number":
        coloring = greedy_color(G, strategy="largest_first")
        return float(max(coloring.values(), default=-1) + 1)

    if feature_name == "clique_number":
        # compute the size of the largest clique by enumerating maximal cliques
        max_size = 0
        for clique in find_cliques(G):
            if len(clique) > max_size:
                max_size = len(clique)
        return float(max_size)

    if feature_name == "domination_number":
        dom = nx_dominating_set(G)
        return float(len(dom))

    if feature_name == "clique_cover_number":
        Gc = nx.complement(G)
        coloring = greedy_color(Gc, strategy="largest_first")
        return float(max(coloring.values(), default=-1) + 1)

    if feature_name == "articulation_points":
        return float(sum(1 for _ in articulation_points(G)))

    # Should never get here
    raise RuntimeError("Unhandled feature")
