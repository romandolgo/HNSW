"""Проверки инвариантов графа.

Заглушки: по мере готовности убирай skip с очередного теста.
"""

import pytest


@pytest.mark.skip(reason="TODO")
def test_euclidean_matches_numpy() -> None:
    """euclidean совпадает с np.linalg.norm на случайных парах точек."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_search_layer_is_exact_with_large_ef() -> None:
    """При ef >= n обход одного связного слоя находит точного соседа."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_edges_are_symmetric() -> None:
    """Связи двунаправленные: если b в соседях a, то a в соседях b."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_degree_within_bounds() -> None:
    """Степень вершины не превышает m_max (m_max0 на нулевом слое)."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_layer_membership_is_nested() -> None:
    """Вершина слоя l присутствует на всех слоях ниже."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_recall_is_one_with_large_ef() -> None:
    """При достаточно большом ef HNSW совпадает с полным перебором."""
    raise NotImplementedError


@pytest.mark.skip(reason="TODO")
def test_level_distribution_matches_m_l() -> None:
    """Среднее число слоёв у вершины близко к 1 / (1 - exp(-1 / m_l)).

    Формула (1) и следующий за ней абзац раздела 4.1. При рекомендованном
    m_l = 1 / ln(M) выражение сворачивается в M / (M - 1).
    """
    raise NotImplementedError
