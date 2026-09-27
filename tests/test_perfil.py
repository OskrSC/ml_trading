import importlib

from mltrading.config import settings
from mltrading.core.env import diagnostics as d


def test_el_perfil_por_defecto_es_ligero():
    assert settings.PERFIL == "ligero"
    assert d.informacion_entorno()["perfil"] == "ligero"


def test_un_perfil_desconocido_vuelve_a_ligero(monkeypatch):
    monkeypatch.setenv("MLT_PERFIL", "otro")
    try:
        recargado = importlib.reload(settings)
        assert recargado.PERFIL == "ligero"
        monkeypatch.setenv("MLT_PERFIL", "completo")
        assert importlib.reload(settings).PERFIL == "completo"
    finally:
        monkeypatch.delenv("MLT_PERFIL", raising=False)
        importlib.reload(settings)


def test_el_perfil_ligero_no_exige_tensorflow():
    requisitos = (settings.RAIZ / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "tensorflow" not in requisitos and "keras" not in requisitos
