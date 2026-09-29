"""Проверки одноуровневого NSW."""

import numpy as np
import pytest

import brute
import data
from graph import select_heuristic, select_simple
from nsw import NSW

N = 300
K = 5


@pytest.fixture(scope="module")
def points() -> np.ndarray:
    return data.uniform(N, seed=0)


@pytest.fixture(scope="module")
def unbounded(points) -> NSW:
    """Классический NSW: степени ничем не ограничены, усечения нет."""
    graph = NSW(points, m=6, ef_construction=30)
    graph.build()
    return graph


@pytest.fixture(scope="module")
def bounded(points) -> NSW:
    """С усечением связей — то, что превращает граф в направленный."""
    graph = NSW(points, m=6, ef_construction=30, m_max=12)
    graph.build()
    return graph


def test_every_point_is_inserted(unbounded, points) -> None:
    assert len(unbounded.graph) == len(points)


def test_edges_are_symmetric(unbounded) -> None:
    """Связи двунаправленные, пока не включено усечение.

    При m_max=None списки соседей никто не пересобирает, поэтому обратная
    ссылка всегда на месте. С усечением это уже не так — см. соседний тест.
    """
    for vertex, neighbours in unbounded.graph.items():
        for neighbour in neighbours:
            assert vertex in unbounded.graph[neighbour]


def test_shrinking_breaks_symmetry(bounded) -> None:
    """Усечение соседа выбрасывает из его списка только что добавленную
    вершину, а обратная ссылка остаётся. Односторонние рёбра — штатное
    поведение HNSW, а не ошибка, и именно поэтому test_edges_are_symmetric
    проверяет только вариант без усечения.
    """
    at_cap = sum(1 for ns in bounded.graph.values() if len(ns) == bounded.m_max)
    assert at_cap > 0, "усечение ни разу не сработало — тест ничего не проверяет"

    one_way = sum(
        1
        for vertex, neighbours in bounded.graph.items()
        for neighbour in neighbours
        if vertex not in bounded.graph[neighbour]
    )
    assert one_way > 0


def test_degree_within_bounds(bounded) -> None:
    """Степень вершины не превышает m_max."""
    assert max(len(ns) for ns in bounded.graph.values()) <= bounded.m_max


def test_no_self_loops_and_no_duplicates(bounded) -> None:
    for vertex, neighbours in bounded.graph.items():
        assert vertex not in neighbours
        assert len(neighbours) == len(set(neighbours))


def test_recall_is_one_with_large_ef(unbounded, points) -> None:
    """При ef=n обход связного графа вырождается в полный перебор."""
    queries = data.uniform(10, seed=1)
    truth = brute.knn_batch(points, queries, K)
    for query, exact in zip(queries, truth, strict=True):
        found = [i for _, i in unbounded.search(query, K, ef=N)]
        assert found == exact.tolist()


def test_search_does_not_mutate_the_index(bounded, points) -> None:
    before = bounded.entry_point, {v: list(ns) for v, ns in bounded.graph.items()}
    for query in data.uniform(20, seed=2):
        bounded.search(query, K, ef=20)
    after = bounded.entry_point, {v: list(ns) for v, ns in bounded.graph.items()}
    assert before == after


def test_repeated_query_gives_the_same_answer(bounded) -> None:
    query = np.array([0.5, 0.5])
    assert bounded.search(query, K, ef=20) == bounded.search(query, K, ef=20)


def test_trace_does_not_change_the_result(bounded) -> None:
    query = np.array([0.5, 0.5])
    trace: list[int] = []
    assert bounded.search(query, K, ef=20, trace=trace) == bounded.search(
        query, K, ef=20
    )
    assert trace


def test_fewer_points_than_k_is_not_an_error(points) -> None:
    """Если в графе меньше k вершин, возвращается всё, что есть."""
    small = NSW(points[:3], m=6, ef_construction=30)
    small.build()
    assert len(small.search(np.array([0.5, 0.5]), 10, ef=10)) == 3


def test_build_twice_raises(points) -> None:
    graph = NSW(points[:50], m=6, ef_construction=30)
    graph.build()
    with pytest.raises(RuntimeError):
        graph.build()


def test_search_before_build_raises(points) -> None:
    with pytest.raises(RuntimeError):
        NSW(points, m=6, ef_construction=30).search(points[0], K, ef=20)


def test_ef_below_k_raises(bounded) -> None:
    with pytest.raises(ValueError):
        bounded.search(np.array([0.5, 0.5]), K, ef=K - 1)


def test_heuristic_keeps_clustered_data_reachable() -> None:
    """Главный результат работы: на кластерах Alg. 3 рвёт связность, Alg. 4 нет.

    Простой отбор соединяет вершину с ближайшими, то есть со своими же по
    сгустку, и мосты между кластерами не возникают. Эвристика строит
    relative neighborhood graph, который глобальную компоненту сохраняет.
    """
    clustered = data.clusters(400, n_clusters=8, sigma=0.01, seed=3)

    def reachable(graph: NSW) -> int:
        seen, stack = {graph.entry_point}, [graph.entry_point]
        while stack:
            for neighbour in graph.graph[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen)

    simple = NSW(clustered, m=8, ef_construction=40, m_max=16, selector=select_simple)
    simple.build()
    heuristic = NSW(
        clustered, m=8, ef_construction=40, m_max=16, selector=select_heuristic
    )
    heuristic.build()

    assert reachable(heuristic) == len(clustered)
    assert reachable(simple) < len(clustered)
