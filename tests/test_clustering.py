"""K-Means, clustering jerárquico, PCA y t-SNE (capítulos 17, 18 y 19)."""
import numpy as np
import pandas as pd
import pytest

from mltrading.core.clustering import hierarchical, kmeans, pca_analysis as pca
from mltrading.core.data.loaders import cargar


@pytest.fixture(scope="module")
def sample_stocks():
    return cargar("sample_stocks.csv")[0]


@pytest.fixture(scope="module")
def pca_precios():
    return cargar("pca.csv")[0]


@pytest.fixture(scope="module")
def capitalizacion():
    df = cargar("stock_list.csv")[0]
    return df.set_index("Symbols")["marketcap"]


# ------------------------------------------------------------------ K-Means ---
def test_kmeans_predeterminado_reproducible(sample_stocks):
    r1 = kmeans.ajustar(sample_stocks, kmeans.K_PREDETERMINADO)
    r2 = kmeans.ajustar(sample_stocks, kmeans.K_PREDETERMINADO)
    assert (r1.etiquetas.values == r2.etiquetas.values).all()
    assert round(r1.inercia, 3) == 7.703
    assert set(r1.etiquetas.unique()) == {0, 1}


def test_kmeans_agrupa_igual_que_el_jerarquico(sample_stocks):
    """Con datos tan separados, ambos métodos deberían coincidir en la partición de 2 grupos."""
    km = kmeans.ajustar(sample_stocks, 2)
    jer = hierarchical.ajustar(sample_stocks, 2)
    coinciden = (km.etiquetas.values == jer.etiquetas.values).all()
    invertidos = (km.etiquetas.values == 1 - jer.etiquetas.values).all()
    assert coinciden or invertidos


def test_kmeans_curva_del_codo_es_decreciente(sample_stocks):
    tabla = kmeans.curva_codo(sample_stocks)
    assert list(tabla["k"]) == list(range(1, 10))
    assert (tabla["inercia"].diff().dropna() <= 0).all()


def test_kmeans_codo_escalado_difiere_del_no_escalado(sample_stocks):
    sin_escalar = kmeans.curva_codo(sample_stocks, escalar_datos=False)
    escalado = kmeans.curva_codo(sample_stocks, escalar_datos=True)
    assert not np.allclose(sin_escalar["inercia"].values, escalado["inercia"].values)


def test_kmeans_valida_k(sample_stocks):
    with pytest.raises(ValueError):
        kmeans.ajustar(sample_stocks, 0)
    with pytest.raises(ValueError):
        kmeans.ajustar(sample_stocks, len(sample_stocks) + 1)


# -------------------------------------------------------------- jerárquico ---
def test_jerarquico_predeterminado_reproducible(sample_stocks):
    r1 = hierarchical.ajustar(sample_stocks, hierarchical.N_CLUSTERS_PREDETERMINADO)
    r2 = hierarchical.ajustar(sample_stocks, hierarchical.N_CLUSTERS_PREDETERMINADO)
    assert (r1.etiquetas.values == r2.etiquetas.values).all()
    assert set(r1.etiquetas.unique()) == {0, 1}


def test_matriz_de_enlace_sin_escalar_por_defecto(sample_stocks):
    from scipy.cluster.hierarchy import linkage

    enlace_modulo = hierarchical.matriz_enlace(sample_stocks)
    enlace_directo = linkage(sample_stocks.values, method="ward")
    np.testing.assert_allclose(enlace_modulo, enlace_directo)


def test_matriz_de_enlace_escalada_difiere(sample_stocks):
    sin_escalar = hierarchical.matriz_enlace(sample_stocks, escalar_datos=False)
    escalada = hierarchical.matriz_enlace(sample_stocks, escalar_datos=True)
    assert not np.allclose(sin_escalar, escalada)


def test_jerarquico_valida_n_clusters(sample_stocks):
    with pytest.raises(ValueError):
        hierarchical.ajustar(sample_stocks, 0)


# --------------------------------------------------------------------- PCA ---
def test_retornos_diarios(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    assert r.shape == (501, 20)
    assert not r.isna().any().any()


def test_pca_predeterminado(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    resultado = pca.ajustar_pca(r, pca.N_COMPONENTES_PREDETERMINADO)
    assert round(resultado.varianza_explicada_acumulada, 4) == 0.9922
    assert resultado.cargas_escaladas.shape == (20, 18)
    assert len(resultado.varianza_explicada) == 18


def test_pca_kmeans_predeterminado_reproducible(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    resultado = pca.ajustar_pca(r)
    c1 = pca.agrupar(resultado, r.columns, pca.K_PREDETERMINADO)
    c2 = pca.agrupar(resultado, r.columns, pca.K_PREDETERMINADO)
    assert (c1.etiquetas.values == c2.etiquetas.values).all()
    assert set(c1.etiquetas.unique()) == {0, 1, 2, 3}
    assert round(c1.inercia, 1) == 286.7


def test_pca_kmeans_etiquetas_conocidas(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    resultado = pca.ajustar_pca(r)
    c1 = pca.agrupar(resultado, r.columns, pca.K_PREDETERMINADO)
    assert c1.etiquetas["GOOG"] == c1.etiquetas["GOOGL"] == c1.etiquetas["AMZN"]
    assert c1.etiquetas["MDLZ"] == c1.etiquetas["CI"]


def test_perplejidad_se_ajusta_con_pocas_muestras():
    assert pca.perplejidad_efectiva(20, 25) == 19
    assert pca.perplejidad_efectiva(100, 25) == 25
    assert pca.perplejidad_efectiva(5, 25) == 4


def test_tsne_predeterminado_reproducible_y_ajustado(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    resultado = pca.ajustar_pca(r)
    t1 = pca.proyectar_tsne(resultado)
    t2 = pca.proyectar_tsne(resultado)
    np.testing.assert_allclose(t1.coordenadas, t2.coordenadas)
    assert t1.coordenadas.shape == (20, 2)
    assert t1.ajustada
    assert t1.perplejidad_usada == 19


def test_pca_valida_n_componentes(pca_precios):
    r = pca.retornos_diarios(pca_precios)
    with pytest.raises(ValueError):
        pca.ajustar_pca(r, 0)
    with pytest.raises(ValueError):
        pca.ajustar_pca(r, 25)  # más que columnas disponibles


def test_capitalizacion_cubre_las_20_acciones(pca_precios, capitalizacion):
    r = pca.retornos_diarios(pca_precios)
    faltantes = set(r.columns) - set(capitalizacion.index)
    assert not faltantes


# ---------------------------------------------------------- gráfico del dendrograma ---
def test_grafico_dendrograma_no_usa_figure_factory(sample_stocks):
    """Cubre el error 'Module scipy has no attribute array', de una función interna de
    plotly.figure_factory incompatible con las versiones recientes de scipy (ver
    docs/solucion_problemas.md). El dendrograma se dibuja a mano a partir de
    scipy.cluster.hierarchy.dendrogram(no_plot=True), sin pasar por esa función."""
    from mltrading.core.clustering.hierarchical import matriz_enlace
    from mltrading.ui.charts import dendrograma

    enlace = matriz_enlace(sample_stocks)
    fig = dendrograma(enlace, list(sample_stocks.index))
    assert len(fig.data) == len(sample_stocks) - 1  # una traza por cada unión del árbol
    assert list(fig.layout.xaxis.ticktext) == list(sample_stocks.index[
        [sample_stocks.index.get_loc(l) for l in fig.layout.xaxis.ticktext]
    ])
    assert set(fig.layout.xaxis.ticktext) == set(sample_stocks.index)
    assert len(fig.layout.xaxis.tickvals) == len(sample_stocks)
    # el rango del eje dejar margen para que la primera y la última etiqueta no se recorten
    assert fig.layout.xaxis.range[0] < min(fig.layout.xaxis.tickvals)
    assert fig.layout.xaxis.range[1] > max(fig.layout.xaxis.tickvals)


def test_grafico_dendrograma_no_importa_scipy_como_modulo_completo():
    """El módulo scipy en sí no necesita exponer .array; solo se usa scipy.cluster.hierarchy."""
    import scipy

    assert not hasattr(scipy, "array")  # confirma que el entorno de pruebas reproduce el caso real
    from mltrading.ui.charts import dendrograma  # no debe fallar al importarse ni al definirse

    assert callable(dendrograma)


def test_precision_temporal_distingue_acierto_y_fallo_sin_depender_solo_del_color():
    import pandas as pd

    from mltrading.ui.charts import precision_temporal

    aciertos = pd.Series([True, False, True, True, False], index=pd.RangeIndex(5))
    fig = precision_temporal(aciertos)
    nombres = {tr.name for tr in fig.data}
    assert nombres == {"Acierto", "Fallo"}
    simbolos = {tr.name: tr.marker.symbol for tr in fig.data}
    assert simbolos["Acierto"] != simbolos["Fallo"]
    ys = {tr.name: set(tr.y) for tr in fig.data}
    assert ys["Acierto"] != ys["Fallo"]  # también se distinguen por posición, no solo color
