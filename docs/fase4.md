# Fase 4: aprendizaje no supervisado

## Alcance

- Página K-Means (capítulo 17): agrupa 12 acciones por ROE y Beta.
- Página Clustering jerárquico (capítulo 18): dendrograma y `AgglomerativeClustering` sobre los mismos datos.
- Página PCA y t-SNE (capítulo 19): reduce 20 acciones a 18 componentes principales, las agrupa con K-Means
  sobre las cargas y las proyecta en 2D con t-SNE, con el tamaño de cada punto según su capitalización.

## Decisión: semillas añadidas donde el código de origen no las fija

El capítulo 17 no fija una semilla para `KMeans`. Verifiqué que, sin semilla, la inercia es siempre la misma
pero las etiquetas 0 y 1 pueden intercambiarse entre ejecuciones (mismo agrupamiento, números distintos). Para
que el resultado predeterminado de esta aplicación sea exactamente reproducible, se añadió `random_state=42`
como valor propio de la aplicación, documentado como tal. El capítulo 19 sí fija semillas (`random_state=7`
para K-Means, `random_state=1337` para t-SNE), así que ahí no hizo falta añadir nada.
`AgglomerativeClustering` no tiene componente aleatoria: es determinista sin necesidad de semilla.

## Decisión: escalado distinto para el ajuste y para las curvas auxiliares

El código de origen ajusta los modelos que se visualizan (`KMeans`, `AgglomerativeClustering`) con los datos
escalados, pero calcula la curva del codo y el dendrograma con los datos sin escalar. Esta aplicación reproduce
esa diferencia: el ajuste del modelo siempre escala, y la curva del codo y el dendrograma tienen una casilla
«Escalar los datos», desmarcada por defecto para reproducir el resultado del código de origen.

## Decisión: perplejidad de t-SNE ajustada

El código de origen usa perplejidad 25 con 20 acciones. La versión de scikit-learn de esta aplicación exige que
la perplejidad sea menor que el número de muestras, así que con 20 acciones el máximo es 19. La aplicación
ajusta la perplejidad solicitada a `min(solicitada, n_muestras - 1)` y lo informa con un aviso visible, sin
ocultar el ajuste.

## Decisión: dendrograma con Plotly, sin matplotlib

El código de origen dibuja el dendrograma con matplotlib. Esta aplicación no incluye matplotlib entre sus
dependencias (todos los gráficos son de Plotly, interactivos y con el tema de la aplicación). Se usa
`plotly.figure_factory.create_dendrogram`, pasándole la matriz de enlace ya calculada con `scipy` en lugar de
dejar que la recalcule, para no duplicar el cálculo.

## Resultados predeterminados

| Elemento | Resultado |
|---|---|
| K-Means (k=2), inercia | 7,70 |
| K-Means y jerárquico | Coinciden en la partición de 2 grupos |
| PCA, 18 componentes, varianza explicada acumulada | 99,22 % |
| PCA + K-Means (k=4), inercia | 286,7 |
| t-SNE, perplejidad solicitada y usada | 25 solicitada, 19 usada |

Todos estos valores, y la reproducibilidad exacta de cada modelo con su semilla, se comprueban en
`tests/test_clustering.py`.

## Módulos nuevos

```
mltrading/core/clustering/
  escalado.py       StandardScaler compartido
  kmeans.py         K-Means (capítulo 17), con semilla añadida
  hierarchical.py   dendrograma y AgglomerativeClustering (capítulo 18)
  pca_analysis.py   PCA, K-Means sobre las cargas y t-SNE (capítulo 19)
mltrading/views/
  kmeans_view.py       página de K-Means
  jerarquico_view.py   página de clustering jerárquico
  pca_view.py          página de PCA y t-SNE
```

Los gráficos de dispersión con clústeres, la curva del codo, la varianza explicada y el dendrograma se
añadieron a `mltrading/ui/charts.py`.
