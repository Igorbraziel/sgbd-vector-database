"""
Restauração de snapshot Arxiv no Qdrant e criação automática de índices.

Usa a API REST diretamente com timeout longo para restaurar o snapshot (9+ GB)
e gerencia a criação de índices de payload no campo 'abstract' para otimizar buscas.
"""

import time
import requests

from qdrant_client.models import TextIndexParams, TextIndexType, TokenizerType
from src.config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, SNAPSHOT_URL
from src.qdrant_client_setup import collection_exists, get_client

# Timeout para a requisição de restore (segundos). O Qdrant baixa e indexa
# o snapshot neste período; 30 min é suficiente para ~9 GB numa boa rede.
_RESTORE_TIMEOUT = 1800


def restore_snapshot() -> str:
    """
    Restaura o snapshot Arxiv no Qdrant e garante a presença dos índices necessários.

    Returns:
        Mensagem de status indicando o resultado.
    """
    client = get_client()
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    collection_already_existed = False
    if collection_exists(COLLECTION_NAME):
        collection_already_existed = True
    else:
        # 1. Restaurar coleção a partir do snapshot
        print(f"⏳ Restaurando snapshot de: {SNAPSHOT_URL}")
        print(f"   Timeout configurado: {_RESTORE_TIMEOUT // 60} min — aguarde...")
        response = requests.put(
            f"{base_url}/collections/{COLLECTION_NAME}/snapshots/recover",
            json={"location": SNAPSHOT_URL},
            timeout=_RESTORE_TIMEOUT,
        )
        response.raise_for_status()

        # Aguarda a coleção ficar pronta (status green)
        print("   Download concluído. Aguardando indexação inicial da coleção...")
        _wait_for_collection(base_url)

    # 2. Verificar e criar índice de texto no campo 'abstract'
    print("⏳ Verificando índices de payload...")
    info = client.get_collection(COLLECTION_NAME)
    schema = info.payload_schema or {}

    index_created = False
    if "abstract" not in schema:
        print("   Criando índice de texto no campo 'abstract' (isso pode levar alguns minutos)...")
        try:
            # Solicita a criação do índice com timeout longo no cliente HTTP
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="abstract",
                field_schema=TextIndexParams(
                    type=TextIndexType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True
                ),
                timeout=300  # 5 minutos
            )
            index_created = True
        except Exception as e:
            # Caso ocorra timeout na conexão do cliente, a criação do índice 
            # continuará rodando de forma assíncrona no servidor Qdrant.
            print(f"   Nota: Solicitação inicial retornou status/erro: {e}")
            print("   Monitorando progresso da criação do índice em background...")
            pass

        # Aguarda a conclusão da indexação de todos os pontos
        _wait_for_index(client)
        print("   Índice 'abstract' criado e ativado com sucesso!")

    # Retornar relatório final detalhado
    info = client.get_collection(COLLECTION_NAME)
    if collection_already_existed:
        msg = f"✅ Coleção '{COLLECTION_NAME}' já existia com {info.points_count:,} pontos."
        if index_created:
            msg += " Novo índice de texto no campo 'abstract' criado e ativado!"
        else:
            msg += " Índice no campo 'abstract' já estava ativo."
        return msg
    else:
        return (
            f"✅ Coleção '{COLLECTION_NAME}' restaurada com sucesso! "
            f"{info.points_count:,} artigos do Arxiv carregados. "
            f"Índice de texto no campo 'abstract' ativado."
        )


def _wait_for_collection(base_url: str, poll_interval: int = 10) -> None:
    """Aguarda até a coleção estar com status 'green'."""
    while True:
        try:
            resp = requests.get(
                f"{base_url}/collections/{COLLECTION_NAME}",
                timeout=30,
            )
            if resp.ok:
                status = resp.json().get("result", {}).get("status", "")
                if status == "green":
                    return
                print(f"   Status atual: {status} — aguardando...")
        except requests.RequestException:
            pass
        time.sleep(poll_interval)


def _wait_for_index(client, poll_interval: int = 5) -> None:
    """Aguarda até que todos os pontos da coleção estejam indexados no campo 'abstract'."""
    while True:
        try:
            info = client.get_collection(COLLECTION_NAME)
            schema = info.payload_schema or {}
            if "abstract" in schema:
                indexed_points = schema["abstract"].points
                total_points = info.points_count
                print(f"   Indexando 'abstract': {indexed_points:,} / {total_points:,} pontos...")
                
                # O índice está pronto se cobrir todos os pontos do dataset
                if indexed_points >= total_points:
                    return
            time.sleep(poll_interval)
        except Exception as e:
            print(f"   Aguardando índice (erro temporário de rede: {e})...")
            time.sleep(poll_interval)


if __name__ == "__main__":
    result = restore_snapshot()
    print(result)
