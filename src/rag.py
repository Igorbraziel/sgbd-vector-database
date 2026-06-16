"""
Pipeline RAG (Retrieval-Augmented Generation).

Fluxo completo:
  1. Pergunta do usuário → embedding InstructorXL
  2. Busca semântica/híbrida no Qdrant → artigos relevantes
  3. Contexto dos artigos → prompt para o LLM (Google Gemini)
  4. Resposta do LLM citando os artigos encontrados

O dataset arxiv_abstracts contém apenas dois campos de payload:
  - abstract: texto do resumo
  - doi:      identificador do artigo

Cadeia de fallback de modelos:
  Gemma 4 27B → Gemini 3.1 Flash Lite → Gemini 2.5 Flash Lite → Gemini 2.5 Flash
"""

from dataclasses import dataclass

from google import genai
from google.genai import types

from src.config import GEMINI_MODELS, GOOGLE_API_KEY
from src.search import ResultadoBusca, busca_semantica, busca_hibrida


@dataclass
class RespostaRAG:
    """Resposta do pipeline RAG."""

    resposta: str
    fontes: list[ResultadoBusca]
    modelo: str
    pergunta: str


def _construir_contexto(artigos: list[ResultadoBusca]) -> str:
    """Formata os artigos encontrados como contexto para o LLM."""
    partes = []
    for i, artigo in enumerate(artigos, 1):
        parte = (
            f"--- Artigo {i} (Similaridade: {artigo.score:.4f}) ---\n"
            f"Resumo: {artigo.resumo}\n"
        )
        if artigo.doi:
            parte += f"DOI: {artigo.doi}\n"
        partes.append(parte)

    return "\n".join(partes)


# Instrução de sistema para o LLM
_SYSTEM_INSTRUCTION = (
    "Você é um assistente acadêmico especializado. "
    "Responda a pergunta baseando-se EXCLUSIVAMENTE no contexto "
    "fornecido (resumos de artigos do Arxiv). "
    "Cite trechos relevantes dos resumos quando usar informações deles. "
    "Se o contexto não contiver informação suficiente, diga isso. "
    "Responda em português brasileiro."
)


def _get_client() -> genai.Client:
    """Cria um cliente Google GenAI."""
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY não configurada. "
            "Obtenha uma chave gratuita em: https://aistudio.google.com/apikey "
            "e adicione ao arquivo .env"
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def _chamar_com_fallback(
    client: genai.Client,
    pergunta: str,
    contexto: str,
) -> tuple[str, str]:
    """
    Tenta gerar resposta usando a cadeia de fallback de modelos.

    Tenta cada modelo na ordem definida em GEMINI_MODELS.
    Se um modelo falhar (quota, indisponível, etc.), tenta o próximo.

    Returns:
        Tupla (resposta_texto, nome_do_modelo_usado).

    Raises:
        RuntimeError: Se todos os modelos falharem.
    """
    prompt = (
        f"Contexto — Resumos de artigos acadêmicos do Arxiv:\n\n"
        f"{contexto}\n\n"
        f"Pergunta: {pergunta}\n\n"
        f"Responda de forma clara e cite os artigos relevantes:"
    )

    erros = []

    for modelo in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=modelo,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=2048,
                ),
            )
            return response.text, modelo

        except Exception as e:
            erros.append(f"  • {modelo}: {e}")
            continue

    # Se chegou aqui, todos os modelos falharam
    erros_str = "\n".join(erros)
    raise RuntimeError(
        f"Todos os modelos falharam:\n{erros_str}\n\n"
        "Verifique sua GOOGLE_API_KEY e conexão com a internet."
    )


def perguntar(
    pergunta: str,
    keyword: str | None = None,
    num_artigos: int = 3,
) -> RespostaRAG:
    """
    Pipeline RAG completo: busca + geração.

    Args:
        pergunta: A pergunta do usuário em linguagem natural.
        keyword: Palavra-chave opcional para filtrar pelo campo 'abstract'.
        num_artigos: Número de artigos a recuperar do Qdrant.

    Returns:
        RespostaRAG com a resposta do LLM e os artigos-fonte.
    """
    # 1. Recuperar artigos relevantes do Qdrant
    if keyword:
        artigos = busca_hibrida(pergunta, keyword=keyword, limite=num_artigos)
    else:
        artigos = busca_semantica(pergunta, limite=num_artigos)

    if not artigos:
        return RespostaRAG(
            resposta="Nenhum artigo relevante encontrado no Qdrant para a sua pergunta.",
            fontes=[],
            modelo="N/A",
            pergunta=pergunta,
        )

    # 2. Construir contexto
    contexto = _construir_contexto(artigos)

    # 3. Chamar Gemini com fallback
    client = _get_client()
    resposta_texto, modelo_usado = _chamar_com_fallback(client, pergunta, contexto)

    return RespostaRAG(
        resposta=resposta_texto,
        fontes=artigos,
        modelo=modelo_usado,
        pergunta=pergunta,
    )
