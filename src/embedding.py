"""
Wrapper para o modelo de embeddings InstructorXL.

Carrega o modelo InstructorXL (hkunlp/instructor-xl) que gera vetores de 768 dimensões,
compatíveis com o snapshot Arxiv pré-vetorizado do Qdrant.

O modelo é carregado uma única vez (singleton) e reutilizado em todas as chamadas.
"""

from InstructorEmbedding import INSTRUCTOR

from src.config import EMBEDDING_MODEL

_model: INSTRUCTOR | None = None


def _get_model() -> INSTRUCTOR:
    """Carrega o modelo InstructorXL (singleton)."""
    global _model
    if _model is None:
        print(f"⏳ Carregando modelo de embeddings: {EMBEDDING_MODEL}...")
        _model = INSTRUCTOR(EMBEDDING_MODEL)
        print("✅ Modelo carregado com sucesso!")
    return _model


def encode(text: str, instruction: str = "Represent the academic document for retrieval:") -> list[float]:
    """
    Gera um vetor de embedding para o texto fornecido.

    Args:
        text: O texto a ser vetorizado.
        instruction: Instrução para o modelo InstructorXL que define o contexto
                     de como o embedding deve ser gerado.

    Returns:
        Lista de 768 floats representando o vetor semântico.
    """
    model = _get_model()
    embeddings = model.encode([[instruction, text]])
    return embeddings[0].tolist()


def encode_query(query: str) -> list[float]:
    """
    Gera um embedding otimizado para consultas de busca.

    Usa uma instrução diferente do encode() para melhor performance
    em tarefas de recuperação (retrieval).

    Args:
        query: A pergunta ou consulta do usuário.

    Returns:
        Lista de 768 floats.
    """
    return encode(
        text=query,
        instruction="Represent the question for retrieving relevant academic papers:",
    )
