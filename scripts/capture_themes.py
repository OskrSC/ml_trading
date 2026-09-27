"""Captura cada página en tema claro y oscuro, en escritorio y móvil.

Requiere Playwright con Chromium instalado. Uso:  python scripts/capture_themes.py
Las imágenes se guardan en docs/capturas/. Sin conexión a Internet, las fuentes de
Google no se descargan y el navegador usa las fuentes de respaldo.
"""
from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "docs" / "capturas"

PAGINAS = {"diagnostico": 2100}
VISTAS = {"escritorio": (1366, 900), "movil": (390, 844)}


def _puerto_libre() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _esperar(url: str, intentos: int = 60) -> None:
    for _ in range(intentos):
        try:
            urllib.request.urlopen(url + "/_stcore/health", timeout=1)
            return
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    raise RuntimeError("El servidor de Streamlit no arrancó a tiempo.")


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    puerto = _puerto_libre()
    servidor = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.headless", "true",
         "--server.port", str(puerto)],
        cwd=RAIZ, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://localhost:{puerto}"
    try:
        _esperar(base)
        with sync_playwright() as p:
            navegador = p.chromium.launch()
            for tema in ("light", "dark"):
                for vista, (ancho, alto) in VISTAS.items():
                    contexto = navegador.new_context(color_scheme=tema, viewport={"width": ancho, "height": alto})
                    for pagina, alto_total in PAGINAS.items():
                        pg = contexto.new_page()
                        pg.set_viewport_size({"width": ancho, "height": alto_total if vista == "escritorio" else alto_total + 1200})
                        pg.goto(f"{base}/{pagina}")
                        pg.wait_for_selector('[data-testid="stMainBlockContainer"]', timeout=30000)
                        pg.wait_for_timeout(4500)
                        pg.screenshot(path=str(DESTINO / f"{pagina or 'inicio'}_{tema}_{vista}.png"))
                        pg.close()
                    contexto.close()
            navegador.close()
    finally:
        servidor.terminate()
    print(f"Capturas en {DESTINO}")


if __name__ == "__main__":
    main()
