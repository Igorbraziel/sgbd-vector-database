# ============================================================
#  Makefile — SGBD Vector Database (Arxiv RAG Demo)
# ============================================================
#  Uso: make <target>
#  Execute 'make' ou 'make help' para ver todos os comandos.
# ============================================================

.DEFAULT_GOAL := help
SHELL := /bin/bash
export PYTHONPATH := .

# --- Variáveis -------------------------------------------------
COMPOSE     := docker compose
APP_SERVICE := app
COLLECTION  := arxiv_papers
QDRANT_URL  := http://localhost:6333

# ============================================================
#  Instalação & Dependências
# ============================================================

.PHONY: install
install: ## Instalar todas as dependências com uv
	uv sync

.PHONY: install-dev
install-dev: ## Instalar dependências incluindo dev (ruff, etc.)
	uv sync --all-extras

.PHONY: lock
lock: ## Regenerar o uv.lock
	uv lock

# ============================================================
#  Desenvolvimento Local
# ============================================================

.PHONY: app
app: ## Executar a UI Streamlit localmente
	uv run streamlit run src/app.py

.PHONY: init
init: ## Restaurar o snapshot Arxiv no Qdrant
	$(COMPOSE) exec $(APP_SERVICE) uv run python scripts/init_collection.py

# ============================================================
#  Docker
# ============================================================

.PHONY: up
up: env ## Subir todos os serviços (Qdrant + App)
	$(COMPOSE) up --build -d

.PHONY: down
down: ## Parar todos os serviços
	$(COMPOSE) down

.PHONY: restart
restart: down up ## Reiniciar todos os serviços

.PHONY: logs
logs: ## Ver logs dos containers (tempo real)
	$(COMPOSE) logs -f

.PHONY: logs-app
logs-app: ## Ver logs apenas da aplicação
	$(COMPOSE) logs -f $(APP_SERVICE)

.PHONY: qdrant
qdrant: ## Subir apenas o Qdrant (para desenvolvimento local)
	$(COMPOSE) up qdrant -d

.PHONY: ps
ps: ## Mostrar status dos containers
	$(COMPOSE) ps

# ============================================================
#  Verificação & Saúde
# ============================================================

.PHONY: health
health: ## Verificar se o Qdrant está saudável
	@curl -sf $(QDRANT_URL)/healthz && echo " ✅ Qdrant OK" || echo " ❌ Qdrant não responde"

.PHONY: collection-info
collection-info: ## Mostrar informações da coleção Arxiv
	@curl -sf $(QDRANT_URL)/collections/$(COLLECTION) | python3 -m json.tool 2>/dev/null || echo "❌ Coleção não encontrada"

# ============================================================
#  Qualidade de Código
# ============================================================

.PHONY: lint
lint: ## Verificar código com ruff
	uv run ruff check src/ scripts/

.PHONY: format
format: ## Formatar código com ruff
	uv run ruff format src/ scripts/

.PHONY: check
check: lint ## Executar todas as verificações

# ============================================================
#  Limpeza
# ============================================================

.PHONY: clean
clean: ## Limpar caches e arquivos temporários
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .ruff_cache

.PHONY: clean-all
clean-all: clean down ## Limpar tudo (caches + containers + volumes)
	$(COMPOSE) down -v
	rm -rf .venv

# ============================================================
#  Utilitários
# ============================================================

.PHONY: env
env: ## Criar .env a partir do .env.example (se não existir)
	@test -f .env || (cp .env.example .env && echo "📄 .env criado a partir de .env.example — configure GOOGLE_API_KEY")

.PHONY: shell
shell: ## Abrir shell no container da aplicação
	$(COMPOSE) exec $(APP_SERVICE) /bin/bash

# ============================================================
#  Ajuda
# ============================================================

.PHONY: help
help: ## Mostrar esta mensagem de ajuda
	@echo ""
	@echo "  🗄️  SGBD Vector Database — Comandos Disponíveis"
	@echo "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
