#!/usr/bin/env python3
"""
Script CLI para restaurar o snapshot Arxiv no Qdrant.

Uso:
    uv run python scripts/init_collection.py
    # ou
    make init
"""

import sys
import os

# Adicionar o diretório raiz ao path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snapshot_restore import restore_snapshot


def main():
    print("=" * 60)
    print("  🗄️  SGBD Vector Database — Inicialização")
    print("=" * 60)
    print()

    try:
        resultado = restore_snapshot()
        print(resultado)
        print()
        print("=" * 60)
        print("  ✅ Inicialização concluída!")
        print("  Acesse a UI em: http://localhost:8501")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Erro durante a inicialização: {e}")
        print()
        print("Verifique se o Qdrant está rodando:")
        print("  make qdrant")
        print("  make health")
        sys.exit(1)


if __name__ == "__main__":
    main()
