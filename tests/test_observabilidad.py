"""Registro estructurado y su buffer en memoria."""
import logging

from mltrading.core import observabilidad as obs


def test_configurar_es_idempotente():
    logger1 = obs.configurar()
    logger2 = obs.configurar()
    assert logger1 is logger2
    # pytest añade sus propios manejadores de captura al ejecutar la suite completa; solo se
    # comprueba que los dos manejadores propios están presentes y no se duplican al llamar dos veces.
    tipos = [type(h).__name__ for h in logger1.handlers]
    assert tipos.count("StreamHandler") == 1
    assert tipos.count("_ManejadorEnMemoria") == 1


def test_obtener_logger_devuelve_un_hijo_de_mltrading():
    logger = obs.obtener_logger("prueba_modulo")
    assert logger.name == "mltrading.prueba_modulo"


def test_las_entradas_quedan_en_el_buffer():
    obs.limpiar_buffer()
    logger = obs.obtener_logger("prueba_buffer")
    logger.warning("mensaje de prueba %s", 123)
    entradas = obs.ultimas_entradas()
    assert any("mensaje de prueba 123" in e.mensaje for e in entradas)
    assert entradas[-1].nivel == "WARNING"
    assert entradas[-1].origen == "mltrading.prueba_buffer"


def test_el_buffer_tiene_un_tamano_maximo():
    obs.limpiar_buffer()
    logger = obs.obtener_logger("prueba_limite")
    for i in range(obs.TAMANO_BUFFER + 20):
        logger.info("entrada %d", i)
    entradas = obs.ultimas_entradas(obs.TAMANO_BUFFER + 20)
    assert len(entradas) == obs.TAMANO_BUFFER
    assert "entrada %d" % (obs.TAMANO_BUFFER + 19) in entradas[-1].mensaje


def test_limpiar_buffer_lo_deja_vacio():
    logger = obs.obtener_logger("prueba_limpiar")
    logger.error("algo salió mal")
    obs.limpiar_buffer()
    assert obs.ultimas_entradas() == []


def test_ultimas_entradas_respeta_el_limite_pedido():
    obs.limpiar_buffer()
    logger = obs.obtener_logger("prueba_n")
    for i in range(10):
        logger.info("e%d", i)
    assert len(obs.ultimas_entradas(3)) == 3


def test_exception_incluye_la_traza_en_el_mensaje():
    obs.limpiar_buffer()
    logger = obs.obtener_logger("prueba_excepcion")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("fallo controlado")
    entradas = obs.ultimas_entradas()
    assert any("ZeroDivisionError" in e.mensaje for e in entradas)
