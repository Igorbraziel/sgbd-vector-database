# ============================================================
# Dockerfile — Aplicação RAG com Qdrant + Streamlit
# ============================================================
# Usa uv para gerenciamento de dependências (rápido e reproduzível)

FROM python:3.11-slim AS base

# Variáveis de ambiente para otimizar Python + uv
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONPATH=/app

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Diretório de trabalho
WORKDIR /app

# Copiar arquivos de dependência primeiro (cache de camadas Docker)
COPY pyproject.toml uv.lock ./

# Instalar dependências (frozen = usa exatamente o uv.lock, mount = cache persistente)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copiar código-fonte
COPY src/ ./src/
COPY scripts/ ./scripts/

# Porta do Streamlit
EXPOSE 8501

# Executar a aplicação Streamlit
CMD ["uv", "run", "streamlit", "run", "src/app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.headless", "true"]
