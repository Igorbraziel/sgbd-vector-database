"""
Configuração do cliente Qdrant.

Provê uma instância singleton do QdrantClient e utilitários de coleção.
"""

from qdrant_client import QdrantClient

from src.config import QDRANT_HOST, QDRANT_PORT

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Retorna uma instância do QdrantClient."""
    global _client
    if _client is None:
        _client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
        )
    return _client


def collection_exists(collection_name: str) -> bool:
    """Verifica se uma coleção existe no Qdrant."""
    client = get_client()
    collections = client.get_collections().collections
    return any(c.name == collection_name for c in collections)


def get_collection_info(collection_name: str) -> dict:
    """Retorna informações detalhadas sobre uma coleção."""
    client = get_client()
    info = client.get_collection(collection_name)
    return {
        "nome": info.config.params.vectors.size if hasattr(info.config.params.vectors, 'size') else "N/A",
        "distancia": str(info.config.params.vectors.distance) if hasattr(info.config.params.vectors, 'distance') else "N/A",
        "pontos": info.points_count,
        "segmentos": info.segments_count,
        "status": str(info.status),
        "config_hnsw": {
            "m": info.config.hnsw_config.m,
            "ef_construct": info.config.hnsw_config.ef_construct,
        },
    }
