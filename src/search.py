"""
Funções de busca no Qdrant.

Implementa três modos de busca para demonstração:
  1. Busca Semântica Pura   — apenas similaridade vetorial.
  2. Busca Híbrida com Filtro — vetor + filtro de texto no campo 'abstract'.
  3. Busca Multi-Filtro     — combinação de must/must_not por palavras no abstract.

O dataset arxiv_abstracts contém apenas dois campos de payload por ponto:
  - abstract: texto do resumo do artigo.
  - doi: identificador digital do artigo.

Cada modo contrasta com o equivalente relacional SQL para fins da apresentação.
"""

from dataclasses import dataclass

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
)

from src.config import COLLECTION_NAME
from src.embedding import encode_query
from src.qdrant_client_setup import get_client


@dataclass
class ResultadoBusca:
    """Resultado de uma busca no Qdrant."""

    score: float
    resumo: str
    doi: str | None = None


def _parse_resultado(point) -> ResultadoBusca:
    """Converte um ScoredPoint do Qdrant em ResultadoBusca."""
    payload = point.payload or {}
    return ResultadoBusca(
        score=point.score,
        resumo=payload.get("abstract", ""),
        doi=payload.get("doi"),
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
    keyword: str | None = None,
    limite: int = 5,
) -> list[ResultadoBusca]:
    """
    Busca híbrida — vetor + filtro de texto no campo 'abstract'.

    Usa MatchText para verificar se a palavra-chave está presente no texto
    do abstract de cada ponto, combinado com a similaridade vetorial.

    Contraste com SQL Relacional:
        Em um SGBD relacional, seria necessário:
            SELECT * FROM papers
            WHERE abstract LIKE '%keyword%'
            ORDER BY similarity(embedding, query_vector) DESC
            LIMIT 5;

        O otimizador relacional avaliaria o WHERE com um full-text index
        ou sequential scan, depois ordenaria. O Qdrant faz tudo em uma
        única travessia do grafo HNSW, verificando o filtro JSON on-the-fly.

    Args:
        query: Pergunta ou tema em linguagem natural.
        keyword: Palavra-chave para filtrar dentro do campo 'abstract'.
        limite: Número máximo de resultados.

    Returns:
        Lista de resultados filtrados e ordenados por similaridade.
    """
    client = get_client()
    query_vector = encode_query(query)

    # Montar filtro de texto no abstract, se keyword fornecida
    query_filter = None
    if keyword:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="abstract",
                    match=MatchText(text=keyword),
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
    keywords_incluir: list[str] | None = None,
    keywords_excluir: list[str] | None = None,
    limite: int = 5,
) -> list[ResultadoBusca]:
    """
    Busca multi-filtro — combinação complexa de condições de texto no abstract.

    Demonstra as capacidades avançadas de filtragem do Qdrant:
        - must:     todas as palavras-chave devem estar no abstract
        - must_not: nenhuma das palavras-chave pode estar no abstract

    Contraste com SQL:
        SELECT * FROM papers
        WHERE abstract LIKE '%neural%'
          AND abstract LIKE '%graph%'
          AND abstract NOT LIKE '%image%'
        ORDER BY similarity(embedding, query_vector) DESC;

        No Qdrant, isso é avaliado durante a travessia HNSW,
        sem necessidade de um índice B-Tree separado.

    Args:
        query: Pergunta ou tema.
        keywords_incluir: Palavras-chave que devem aparecer no abstract.
        keywords_excluir: Palavras-chave que NÃO podem aparecer no abstract.
        limite: Número máximo de resultados.

    Returns:
        Lista de resultados filtrados.
    """
    client = get_client()
    query_vector = encode_query(query)

    must_conditions = []
    must_not_conditions = []

    if keywords_incluir:
        for kw in keywords_incluir:
            must_conditions.append(
                FieldCondition(key="abstract", match=MatchText(text=kw))
            )

    if keywords_excluir:
        for kw in keywords_excluir:
            must_not_conditions.append(
                FieldCondition(key="abstract", match=MatchText(text=kw))
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
