import pandas as pd
import pytest

from mltrading.config import settings
from mltrading.core.data.catalog import CATALOGO, ORIGINAL, PREDETERMINADO
from mltrading.core.data.loaders import cargar
from mltrading.core.data.validation import AVISO, ERROR, validar

# archivo: (filas, primera fecha, última fecha, filas con nulos)
ESPERADO = {
    "JPM_2017_2019.csv": (19370, "2017-01-03 09:45", "2019-12-31 16:00", 0),
    "JPM_features_2017_2019.csv": (19317, "2017-01-05 09:45", "2019-12-31 15:45", 0),
    "JPM_features_training_2017_2019.csv": (15453, "2017-01-05 09:45", "2019-05-28 11:45", 0),
    "JPM_features_testing_2017_2019.csv": (3864, "2019-05-28 12:00", "2019-12-31 15:45", 0),
    "JPM_target_2017_2019.csv": (19317, "2017-01-05 09:45", "2019-12-31 15:45", 0),
    "JPM_target_training_2017_2019.csv": (15453, "2017-01-05 09:45", "2019-05-28 11:45", 0),
    "JPM_target_testing_2017_2019.csv": (3864, "2019-05-28 12:00", "2019-12-31 15:45", 0),
    "JPM_predicted_2017_2019.csv": (3864, "2019-05-28 12:00", "2019-12-31 15:45", 0),
    "RELIANCE.NS.csv": (5638, "1996-01-01", "2018-04-19", 120),
    "coca_cola_price.csv": (565, "2019-01-02", "2021-03-30", 0),
    "jpm_and_bac_price.csv": (312, "2020-01-03", "2021-03-30", 0),
    "jpm_and_bac_price_2019.csv": (251, "2019-01-02", "2019-12-30", 0),
    "predicted_jpm_and_nestle_price_2019.csv": (251, "2019-01-02", "2019-12-30", 0),
    "pca.csv": (502, "2018-01-02", "2019-12-30", 0),
    "sample_stocks.csv": (12, None, None, 0),
    "stock_list.csv": (89, None, None, 0),
}


def test_el_catalogo_cubre_exactamente_los_csv_de_data_modules():
    en_disco = {p.name for p in settings.DIRECTORIO_DATOS.glob("*.csv")}
    assert {e.archivo for e in CATALOGO} == en_disco
    assert len(CATALOGO) == 16


def test_tipos_del_catalogo():
    tipos = {e.archivo: e.tipo for e in CATALOGO}
    assert sum(t == PREDETERMINADO for t in tipos.values()) == 7
    assert sum(t == ORIGINAL for t in tipos.values()) == 9


@pytest.mark.parametrize("archivo", list(ESPERADO))
def test_lectura_y_normalizacion(archivo):
    filas, inicio, fin, con_nulos = ESPERADO[archivo]
    df, informe = cargar(archivo)
    assert informe.filas == filas == len(df)
    assert informe.filas_con_nulos == con_nulos
    assert informe.indice_duplicados == 0
    if inicio:
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is None
        assert df.index.is_monotonic_increasing
        assert df.index.name == "fecha"
        assert f"{df.index[0]:%Y-%m-%d %H:%M}".startswith(inicio)
        assert f"{df.index[-1]:%Y-%m-%d %H:%M}".startswith(fin)


def test_fechas_con_dia_primero_se_interpretan_bien():
    df, _ = cargar("jpm_and_bac_price_2019.csv")
    assert df.index[0] == pd.Timestamp("2019-01-02")  # 02-01-2019 es el 2 de enero
    assert df.index.max() == pd.Timestamp("2019-12-30")


def test_la_columna_sobrante_de_stock_list_se_descarta():
    df, informe = cargar("stock_list.csv")
    assert list(df.columns) == ["Symbols", "marketcap"]
    assert informe.columnas == 2


def test_solo_reliance_tiene_avisos():
    for spec in CATALOGO:
        df, informe = cargar(spec.archivo)
        graves = [i for i in validar(df, spec, informe) if i.nivel in {AVISO, ERROR}]
        if spec.archivo == "RELIANCE.NS.csv":
            assert [i.codigo for i in graves] == ["nulos"]
        else:
            assert graves == [], (spec.archivo, graves)


def test_descartar_nulos_es_explicito():
    df_completo, _ = cargar("RELIANCE.NS.csv")
    df_limpio, informe = cargar("RELIANCE.NS.csv", descartar_nulos=True)
    assert len(df_completo) == 5638
    assert len(df_limpio) == 5518
    assert informe.filas_descartadas == 120
    assert not df_limpio.isna().any().any()


def test_division_entrenamiento_prueba_es_coherente():
    entrenamiento, _ = cargar("JPM_features_training_2017_2019.csv")
    prueba, _ = cargar("JPM_features_testing_2017_2019.csv")
    completo, _ = cargar("JPM_features_2017_2019.csv")
    predichas, _ = cargar("JPM_predicted_2017_2019.csv")
    assert len(entrenamiento) + len(prueba) == len(completo)
    assert entrenamiento.index.max() < prueba.index.min()
    assert predichas.index.equals(prueba.index)


def test_validacion_detecta_columnas_faltantes():
    spec = next(e for e in CATALOGO if e.archivo == "JPM_2017_2019.csv")
    df, informe = cargar(spec.archivo)
    df = df.drop(columns=["volume"])
    informe.columnas_presentes = tuple(df.columns)
    codigos = [i.codigo for i in validar(df, spec, informe)]
    assert "columnas_faltantes" in codigos
