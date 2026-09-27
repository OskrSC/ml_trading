"""Regenera .streamlit/config.toml desde mltrading/config/tokens.py.

Uso:  python scripts/generate_theme.py
"""
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from mltrading.config.theme_config import generar_config_toml  # noqa: E402

destino = RAIZ / ".streamlit" / "config.toml"
destino.write_text(generar_config_toml(), encoding="utf-8")
print(f"Escrito: {destino}")
