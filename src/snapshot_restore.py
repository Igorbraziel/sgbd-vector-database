"""
Restauração do snapshot Arxiv no Qdrant.

Baixa e restaura a coleção pré-vetorizada com embeddings InstructorXL (768-dim).
O processo é idempotente — não faz nada se a coleção já existir.
"""

from src.config import COLLECTION_NAME, SNAPSHOT_URL
from src.qdrant_client_setup import collection_exists, get_client


def restore_snapshot() -> str:
    """
    Restaura o snapshot Arxiv no Qdrant.

    Returns:
        Mensagem de status indicando o resultado.
    """
    if collection_exists(COLLECTION_NAME):
        client = get_client()
        info = client.get_collection(COLLECTION_NAME)
        return (
            f"✅ Coleção '{COLLECTION_NAME}' já existe com "
            f"{info.points_count:,} pontos. Nenhuma ação necessária."
        )

    client = get_client()

    print(f"⏳ Restaurando snapshot de: {SNAPSHOT_URL}")
    print("   Isso pode levar alguns minutos na primeira execução...")

    client.recover_snapshot(
        collection_name=COLLECTION_NAME,
        location=SNAPSHOT_URL,
    )

    # Verificar resultado
    info = client.get_collection(COLLECTION_NAME)
    return (
        f"✅ Coleção '{COLLECTION_NAME}' restaurada com sucesso! "
        f"{info.points_count:,} artigos do Arxiv carregados."
    )


if __name__ == "__main__":
    result = restore_snapshot()
    print(result)
