"""
Interface Streamlit — Demo RAG com Qdrant.

Quatro abas para demonstração durante a apresentação:
  🔍 Busca Semântica — busca por similaridade vetorial pura
  🎯 Busca Híbrida   — vetor + filtro de texto no campo 'abstract'
  🤖 RAG Q&A         — pipeline completo com LLM (Google Gemini)
  ⚙️ Info da Coleção  — estatísticas, esquema e configuração HNSW

Dataset: arxiv_abstracts (Qdrant public snapshot)
Campos de payload disponíveis: abstract, doi
"""

import streamlit as st

from src.config import COLLECTION_NAME, GEMINI_MODELS
from src.qdrant_client_setup import collection_exists, get_collection_info
from src.search import busca_semantica, busca_hibrida, busca_multi_filtro
from src.rag import perguntar

# ============================================================
#  Configuração da Página
# ============================================================

st.set_page_config(
    page_title="SGBD Vetorial — Demo RAG com Qdrant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CSS Customizado
# ============================================================

st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 1.1rem;
    }

    /* Card de resultado */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
    }
    .result-card .score {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .result-card .doi-badge {
        background: rgba(102, 126, 234, 0.15);
        color: #a5b4fc;
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-family: monospace;
        margin-left: 0.5rem;
        display: inline-block;
    }
    .result-card .abstract-text {
        color: rgba(255,255,255,0.75);
        font-size: 0.92rem;
        line-height: 1.65;
        margin-top: 0.8rem;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box h4 {
        color: #667eea;
    }

    /* Métricas */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .label {
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* RAG response */
    .rag-response {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    /* Keyword highlight badge */
    .keyword-badge {
        background: rgba(118, 75, 162, 0.25);
        color: #c084fc;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-family: monospace;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
#  Header
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🗄️ SGBD Vetorial — Qdrant</h1>
    <p>Demo de busca semântica e RAG com artigos do Arxiv</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
#  Verificação de Conexão
# ============================================================

def _check_connection():
    """Verifica se o Qdrant está acessível e a coleção existe."""
    try:
        if not collection_exists(COLLECTION_NAME):
            st.warning(
                f"⚠️ Coleção '{COLLECTION_NAME}' não encontrada. "
                f"Execute `make init` para restaurar o snapshot."
            )
            return False
        return True
    except Exception as e:
        st.error(f"❌ Erro ao conectar ao Qdrant: {e}")
        return False


# ============================================================
#  Funções Auxiliares de Renderização
# ============================================================

def _render_resultado(resultado, idx: int, highlight_keyword: str | None = None):
    """Renderiza um card de resultado de busca."""
    doi_html = (
        f'<span class="doi-badge">DOI: {resultado.doi}</span>'
        if resultado.doi else ""
    )

    # Score badge + DOI (styled HTML, no user-generated content)
    st.markdown(f"""
    <div class="result-card">
        <span class="score">🎯 Score: {resultado.score:.4f}</span>
        {doi_html}
    </div>
    """, unsafe_allow_html=True)

    # Abstract text using native Streamlit (safe for LaTeX/special chars)
    resumo_preview = resultado.resumo[:400].strip()
    if len(resultado.resumo) > 400:
        resumo_preview += "..."

    st.caption(resumo_preview)

    # Full abstract in expander if truncated
    if len(resultado.resumo) > 400:
        with st.expander(f"📄 Ver resumo completo — resultado #{idx + 1}"):
            st.write(resultado.resumo)


# ============================================================
#  Abas
# ============================================================

tab_semantica, tab_hibrida, tab_rag, tab_info = st.tabs([
    "🔍 Busca Semântica",
    "🎯 Busca Híbrida",
    "🤖 RAG Q&A",
    "⚙️ Info da Coleção",
])


# --- Aba 1: Busca Semântica ---
with tab_semantica:
    st.markdown("### 🔍 Busca Semântica Pura")
    st.markdown(
        "Encontra artigos por **similaridade vetorial** — sem filtros, "
        "apenas a semântica da sua pergunta."
    )

    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            query_sem = st.text_input(
                "Sua pergunta:",
                placeholder="Ex: machine learning for database optimization",
                key="query_semantica",
            )
        with col2:
            limite_sem = st.number_input("Resultados:", min_value=1, max_value=20, value=5, key="limite_sem")

    if st.button("🔍 Buscar", key="btn_semantica", type="primary"):
        if query_sem and _check_connection():
            with st.spinner("Gerando embedding e buscando no Qdrant..."):
                resultados = busca_semantica(query_sem, limite=limite_sem)

            if resultados:
                st.success(f"✅ {len(resultados)} artigos encontrados!")
                for i, r in enumerate(resultados):
                    _render_resultado(r, i)
            else:
                st.info("Nenhum resultado encontrado.")

    with st.expander("💡 Contraste com SGBD Relacional"):
        st.markdown("""
        **No Qdrant (SGBD Vetorial):**
        - A consulta é convertida em um vetor de 768 dimensões
        - O índice HNSW encontra os vizinhos mais próximos em tempo sub-linear `O(log n)`
        - Não precisa de palavras-chave exatas

        **Em um SGBD Relacional (ex: PostgreSQL sem pgvector):**
        - Seria necessário `LIKE '%keyword%'` ou full-text search (tsvector)
        - Busca por similaridade semântica é **impossível** sem extensão vetorial
        - Full-text search não entende sinônimos ou contexto
        """)


# --- Aba 2: Busca Híbrida ---
with tab_hibrida:
    st.markdown("### 🎯 Busca Híbrida (Vetor + Filtro de Texto)")
    st.markdown(
        "Combina **similaridade vetorial** com **filtro de palavra-chave no abstract** "
        "em uma única operação — o grande diferencial do Qdrant."
    )
    st.info(
        "💡 **Como funciona:** a palavra-chave é verificada diretamente no campo `abstract` "
        "de cada ponto usando `MatchText`, enquanto o HNSW navega o grafo de vizinhos. "
        "Nenhuma etapa separada de filtragem.",
        icon="🔑",
    )

    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        with col1:
            query_hib = st.text_input(
                "Sua pergunta:",
                placeholder="Ex: approximate nearest neighbor search algorithms",
                key="query_hibrida",
            )
        with col2:
            keyword_hib = st.text_input(
                "Filtrar por palavra-chave no abstract:",
                placeholder="Ex: transformer, graph, neural",
                key="keyword_hibrida",
            )
        with col3:
            limite_hib = st.number_input("Resultados:", min_value=1, max_value=20, value=5, key="limite_hib")

    if st.button("🎯 Buscar com Filtro", key="btn_hibrida", type="primary"):
        if query_hib and _check_connection():
            kw = keyword_hib.strip() if keyword_hib else None
            with st.spinner("Busca híbrida em andamento..."):
                resultados = busca_hibrida(query_hib, keyword=kw, limite=limite_hib)

            if resultados:
                kw_label = f' contendo **"{kw}"** no abstract' if kw else ""
                st.success(f"✅ {len(resultados)} artigos encontrados{kw_label}!")
                for i, r in enumerate(resultados):
                    _render_resultado(r, i, highlight_keyword=kw)
            else:
                st.info("Nenhum resultado encontrado com esses filtros.")

    with st.expander("💡 Contraste com SGBD Relacional"):
        st.markdown("""
        **No Qdrant — Filtro em Uma Etapa:**
        ```python
        query_points(
            query=vetor_768d,
            query_filter=Filter(
                must=[FieldCondition(key="abstract", match=MatchText(text="transformer"))]
            )
        )
        ```
        O filtro de texto é verificado **durante** a travessia do grafo HNSW.
        Não há etapa separada de filtragem.

        **Em um SGBD Relacional — Duas Etapas:**
        ```sql
        SELECT * FROM papers
        WHERE abstract LIKE '%transformer%'        -- 1. Full-text scan / índice GIN
        ORDER BY similarity(embedding, ?) DESC     -- 2. Ordenação separada
        LIMIT 5;
        ```
        O otimizador de consultas (cost-based optimizer) decide se usa o índice
        GIN/GiST primeiro ou faz um sequential scan. São **duas operações distintas**.
        """)


# --- Aba 3: RAG Q&A ---
with tab_rag:
    st.markdown("### 🤖 RAG — Perguntas e Respostas")
    st.markdown(
        f"Pipeline completo: sua pergunta → embedding → Qdrant → "
        f"**Google Gemini** → resposta com citações."
    )

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            query_rag = st.text_input(
                "Faça sua pergunta:",
                placeholder="Ex: Quais são as principais técnicas de indexação para busca vetorial?",
                key="query_rag",
            )
        with col2:
            kw_rag = st.text_input(
                "Palavra-chave no abstract (opcional):",
                placeholder="Ex: HNSW, attention",
                key="kw_rag",
            )

    num_artigos = st.slider(
        "Número de artigos para contexto:",
        min_value=1, max_value=10, value=3, key="num_artigos_rag",
    )

    if st.button("🤖 Perguntar", key="btn_rag", type="primary"):
        if query_rag and _check_connection():
            kw = kw_rag.strip() if kw_rag else None
            with st.spinner("🔄 Buscando artigos e consultando Google Gemini..."):
                try:
                    resultado_rag = perguntar(
                        pergunta=query_rag,
                        keyword=kw,
                        num_artigos=num_artigos,
                    )

                    # Resposta do LLM
                    st.markdown(f"#### 💬 Resposta (via `{resultado_rag.modelo}`)")
                    st.markdown("---")
                    st.markdown(resultado_rag.resposta)
                    st.markdown("---")

                    # Fontes
                    st.markdown(f"#### 📚 Fontes ({len(resultado_rag.fontes)} artigos)")
                    for i, fonte in enumerate(resultado_rag.fontes):
                        _render_resultado(fonte, i)

                except Exception as e:
                    st.error(f"❌ Erro no pipeline RAG: {e}")
                    st.info(
                        "💡 Verifique se a GOOGLE_API_KEY está configurada no .env. "
                        "Obtenha uma chave gratuita em: https://aistudio.google.com/apikey"
                    )


# --- Aba 4: Info da Coleção ---
with tab_info:
    st.markdown("### ⚙️ Informações da Coleção")
    st.markdown("Detalhes arquiteturais do SGBD vetorial Qdrant.")

    if _check_connection():
        try:
            info = get_collection_info(COLLECTION_NAME)

            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{info['pontos']:,}</div>
                    <div class="label">Pontos (Artigos)</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{info['nome']}</div>
                    <div class="label">Dimensões do Vetor</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{info['distancia']}</div>
                    <div class="label">Métrica de Distância</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">{info['segmentos']}</div>
                    <div class="label">Segmentos</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Payload info
            st.markdown("#### 📦 Campos de Payload Disponíveis")
            st.markdown("""
            <div class="info-box">
                <h4>Dataset: arxiv_abstracts (Qdrant Public Snapshot)</h4>
                <p>Este snapshot pré-computado é otimizado para benchmarking de busca semântica
                e expõe apenas dois campos de payload por ponto:</p>
                <ul>
                    <li><strong>abstract</strong> — Texto do resumo do artigo (usado para gerar o embedding)</li>
                    <li><strong>doi</strong> — Identificador digital do artigo (DOI / arXiv ID)</li>
                </ul>
                <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem;">
                    Campos como título, autores e categorias não estão presentes neste snapshot.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Detalhes HNSW
            st.markdown("#### 🔗 Configuração do Índice HNSW")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-box">
                    <h4>Parâmetro M = {info['config_hnsw']['m']}</h4>
                    <p>Número máximo de conexões por nó no grafo HNSW.
                    Valores maiores = maior precisão, mais memória.</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="info-box">
                    <h4>ef_construct = {info['config_hnsw']['ef_construct']}</h4>
                    <p>Tamanho da lista de candidatos durante a construção do índice.
                    Valores maiores = construção mais lenta, melhor qualidade.</p>
                </div>
                """, unsafe_allow_html=True)

            # Contraste arquitetural
            with st.expander("💡 Contraste Arquitetural com SGBD Relacional"):
                st.markdown("""
                | Aspecto | SGBD Relacional | Qdrant (Vetorial) |
                |---|---|---|
                | **Índice Principal** | B-Tree / Hash | HNSW (grafo navegável) |
                | **Busca** | Exata (O(log n) B-Tree) | Aproximada (ANN, O(log n) HNSW) |
                | **Armazenamento** | Páginas de tamanho fixo | Segmentos com vetores compactados |
                | **Filtros** | WHERE com índice separado | Verificação on-the-fly no HNSW |
                | **Transações** | ACID completo (WAL + MVCC) | WAL simplificado |
                | **Otimizador** | Cost-based (estatísticas) | Heurísticas HNSW (ef, M) |
                | **Modelo de Dados** | Tabelas com esquema fixo | Pontos com payload JSON flexível |
                """)

        except Exception as e:
            st.error(f"❌ Erro ao obter informações: {e}")


# ============================================================
#  Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📖 Sobre")
    st.markdown(
        "Este demo faz parte do trabalho da disciplina "
        "**Sistemas Gerenciadores de Banco de Dados** (UFG)."
    )
    st.markdown("---")
    st.markdown("### 🛠️ Tecnologias")
    st.markdown("""
    - **SGBD Vetorial:** Qdrant
    - **Embeddings:** InstructorXL (768-dim)
    - **LLM:** Google Gemini (com fallback)
    - **Interface:** Streamlit
    - **Orquestração:** Docker Compose
    """)
    st.markdown("---")
    st.markdown("### 📦 Dataset")
    st.markdown("""
    **arxiv_abstracts** (Qdrant snapshot público)

    Campos de payload:
    - `abstract` — resumo do artigo
    - `doi` — identificador do artigo
    """)
    st.markdown("---")
    st.markdown("### 🤖 Cadeia de Modelos")
    st.markdown("Fallback automático se um modelo falhar:")
    for i, model in enumerate(GEMINI_MODELS):
        icon = "1️⃣" if i == 0 else "2️⃣" if i == 1 else "3️⃣" if i == 2 else "4️⃣"
        st.markdown(f"{icon} `{model}`")
    st.markdown("---")
    st.markdown("### 🔗 Links Úteis")
    st.markdown("""
    - [Qdrant Docs](https://qdrant.tech/documentation/)
    - [Arxiv Dataset](https://qdrant.tech/documentation/datasets/)
    - [InstructorXL](https://huggingface.co/hkunlp/instructor-xl)
    - [Google AI Studio](https://aistudio.google.com/)
    """)
