"""
Funções de busca no Qdrant.

Implementa três modos de busca para demonstração:
  1. Busca Semântica Pura — apenas similaridade vetorial.
  2. Busca Híbrida com Filtro — vetor + filtro de metadados (ex: categoria).
  3. Busca Multi-Filtro — combinação de condições (must, should, must_not).

Cada modo contrasta com o equivalente relacional SQL para fins da apresentação.
"""

from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from src.config import COLLECTION_NAME
from src.embedding import encode_query
from src.qdrant_client_setup import get_client


@dataclass
class ResultadoBusca:
    """Resultado de uma busca no Qdrant."""

    score: float
    titulo: str
    resumo: str
    autores: str
    categorias: str
    doi: str | None = None
    journal_ref: str | None = None


def _parse_resultado(point) -> ResultadoBusca:
    """Converte um ScoredPoint do Qdrant em ResultadoBusca."""
    payload = point.payload or {}
    return ResultadoBusca(
        score=point.score,
        titulo=payload.get("title", "Sem título"),
        resumo=payload.get("abstract", ""),
        autores=payload.get("authors", ""),
        categorias=payload.get("categories", ""),
        doi=payload.get("doi"),
        journal_ref=payload.get("journal-ref"),
    )


def busca_semantica(query: str, limite: int = 5) -> list[ResultadoBusca]:
    """
    Busca semântica pura — encontra artigos por similaridade vetorial.

    Equivalente relacional impossível:
        Não existe um SELECT em SQL que capture similaridade semântica
        sem uma extensão vetorial (ex: pgvector).

    Args:
        query: Pergunta ou tema em linguagem natural.
        limite: Número máximo de resultados.

    Returns:
        Lista de resultados ordenados por similaridade.
    """
    client = get_client()
    query_vector = encode_query(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limite,
    )

    return [_parse_resultado(point) for point in results.points]


def busca_hibrida(
    query: str,
    categoria: str | None = None,
    limite: int = 5,
) -> list[ResultadoBusca]:
    """
    Busca híbrida — vetor + filtro de metadados em uma única etapa.

    Contraste com SQL Relacional:
        Em um SGBD relacional, seria necessário:
            SELECT * FROM papers
            WHERE categories = 'cs.DB'
            ORDER BY similarity(embedding, query_vector) DESC
            LIMIT 5;

        O otimizador relacional avaliaria o WHERE com um índice B-Tree
        e depois faria a ordenação. O Qdrant faz tudo em uma única
        travessia do grafo HNSW, verificando o filtro JSON on-the-fly.

    Args:
        query: Pergunta ou tema em linguagem natural.
        categoria: Filtro por categoria Arxiv (ex: "cs.DB", "cs.AI").
        limite: Número máximo de resultados.

    Returns:
        Lista de resultados filtrados e ordenados por similaridade.
    """
    client = get_client()
    query_vector = encode_query(query)

    # Montar filtro se a categoria foi fornecida
    query_filter = None
    if categoria:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="categories",
                    match=MatchValue(value=categoria),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limite,
    )

    return [_parse_resultado(point) for point in results.points]


def busca_multi_filtro(
    query: str,
    categorias_incluir: list[str] | None = None,
    categorias_excluir: list[str] | None = None,
    limite: int = 5,
) -> list[ResultadoBusca]:
    """
    Busca multi-filtro — combinação complexa de condições.

    Demonstra as capacidades avançadas de filtragem do Qdrant:
        - must: todas as condições devem ser verdadeiras
        - must_not: nenhuma das condições pode ser verdadeira

    Contraste com SQL:
        SELECT * FROM papers
        WHERE categories IN ('cs.DB', 'cs.AI')
          AND categories NOT IN ('cs.CV')
        ORDER BY similarity(embedding, query_vector) DESC;

        No Qdrant, isso é avaliado durante a travessia HNSW,
        sem necessidade de um índice B-Tree separado.

    Args:
        query: Pergunta ou tema.
        categorias_incluir: Lista de categorias que devem estar presentes.
        categorias_excluir: Lista de categorias a excluir.
        limite: Número máximo de resultados.

    Returns:
        Lista de resultados filtrados.
    """
    client = get_client()
    query_vector = encode_query(query)

    must_conditions = []
    must_not_conditions = []

    if categorias_incluir:
        for cat in categorias_incluir:
            must_conditions.append(
                FieldCondition(key="categories", match=MatchValue(value=cat))
            )

    if categorias_excluir:
        for cat in categorias_excluir:
            must_not_conditions.append(
                FieldCondition(key="categories", match=MatchValue(value=cat))
            )

    query_filter = None
    if must_conditions or must_not_conditions:
        query_filter = Filter(
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=limite,
    )

    return [_parse_resultado(point) for point in results.points]
