"""
Wrapper para o modelo de embeddings InstructorXL.

Carrega o modelo InstructorXL (hkunlp/instructor-xl) que gera vetores de 768 dimensões,
compatíveis com o snapshot Arxiv pré-vetorizado do Qdrant.

O modelo é carregado uma única vez (singleton) e reutilizado em todas as chamadas.
"""

from InstructorEmbedding import INSTRUCTOR

# Monkey-patch: INSTRUCTOR inherits from SentenceTransformer. Newer versions of
# sentence-transformers have removed/renamed _text_length, which breaks INSTRUCTOR's
# internal batch sorting in encode().
if not hasattr(INSTRUCTOR, "_text_length"):
    def _text_length(self, text):
        if isinstance(text, dict):
            return len(text.get("title", "")) + len(text.get("text", ""))
        elif isinstance(text, (list, tuple)):
            return sum([len(t) for t in text])
        return len(text)
    INSTRUCTOR._text_length = _text_length

from src.config import EMBEDDING_MODEL, GOOGLE_API_KEY, GEMINI_MODELS, HF_TOKEN

_model: INSTRUCTOR | None = None


def _get_model() -> INSTRUCTOR:
    """Carrega o modelo InstructorXL (singleton)."""
    global _model
    if _model is None:
        print(f"⏳ Carregando modelo de embeddings: {EMBEDDING_MODEL}...")
        kwargs = {}
        if HF_TOKEN:
            # Passa o token para evitar rate limits no Hugging Face
            kwargs["token"] = HF_TOKEN
            
        _model = INSTRUCTOR(EMBEDDING_MODEL, **kwargs)
        print("✅ Modelo carregado com sucesso!")
    return _model


def _translate_to_english(text: str) -> str:
    """
    Translates a query to English using Google Gemini.

    The arxiv_abstracts dataset contains only English abstracts, so queries
    in other languages (e.g. Portuguese) produce embeddings that match on
    language signal rather than semantic meaning. Translating to English
    before encoding ensures the query vector aligns with the corpus.

    Falls back to the original text if translation fails.
    """
    if not GOOGLE_API_KEY:
        return text

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GOOGLE_API_KEY)

        for modelo in GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=modelo,
                    contents=(
                        "Translate the following text to English. "
                        "If it is already in English, return it unchanged. "
                        "Return ONLY the translated text, nothing else.\n\n"
                        f"{text}"
                    ),
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=256,
                    ),
                )
                translated = response.text.strip()
                if translated:
                    if translated != text:
                        print(f"🌐 Query traduzida: '{text}' → '{translated}'")
                    return translated
            except Exception:
                continue

    except Exception:
        pass

    return text


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

    Primeiro traduz a consulta para inglês (se necessário), já que o dataset
    contém apenas abstracts em inglês. Em seguida, gera o embedding com uma
    instrução otimizada para tarefas de recuperação (retrieval).

    Args:
        query: A pergunta ou consulta do usuário (em qualquer idioma).

    Returns:
        Lista de 768 floats.
    """
    translated_query = _translate_to_english(query)
    return encode(
        text=translated_query,
        instruction="Represent the question for retrieving relevant academic papers:",
    )
