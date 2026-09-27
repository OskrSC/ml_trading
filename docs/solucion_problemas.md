# Solución de problemas

## ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'

Aparece al importar `statsmodels` (usado en la página Regresión y en la prueba ADF de Variables y objetivo).

**Causa.** `scipy` retiró `_lazywhere` de `scipy._lib._util` en la versión 1.16. Los `statsmodels` anteriores a
la 0.14.5 todavía la importan. El error aparece cuando el entorno tiene un `scipy` reciente (1.16 o superior)
junto con un `statsmodels` más viejo, o cualquier otra combinación incompatible entre los dos.

`requirements.txt` fija `scipy==1.17.1` junto con `statsmodels==0.15.0`, una combinación compatible. Ver este
error significa que el entorno activo no coincide con esas versiones fijadas, casi siempre porque se creó
antes de que existiera ese archivo, o porque se instaló algún paquete suelto sin las versiones fijadas.

**Solución.** Recree el entorno virtual e instale solo desde `requirements.txt`:

```powershell
deactivate
Remove-Item -Recurse -Force <ruta_del_venv>
python -m venv <ruta_del_venv>
<ruta_del_venv>\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

En Linux o macOS, cambie los tres últimos comandos por:

```bash
rm -rf <ruta_del_venv>
python3 -m venv <ruta_del_venv>
source <ruta_del_venv>/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

O, sin recrear el entorno, force la reinstalación de los dos paquetes:

```
pip install --force-reinstall "scipy==1.17.1" "statsmodels==0.15.0"
```

**Verificación:**

```
pip show scipy statsmodels
python -c "import statsmodels.api"
```

`pip show` debe reportar exactamente `scipy 1.17.1` y `statsmodels 0.15.0` (o una combinación más nueva que
ambos proyectos confirmen compatible entre sí). Si el error persiste con las versiones correctas instaladas,
revise si hay más de un intérprete de Python en el `PATH` y confirme con `python -m pip show scipy` que está
inspeccionando el mismo entorno donde corre `streamlit run`.

## ModuleNotFoundError: No module named 'xgboost' (o cualquier otro paquete de requirements.txt)

**Causa.** El entorno activo no tiene instaladas todas las dependencias de `requirements.txt`. Suele pasar
cuando el `venv` se creó antes de que existiera ese archivo, o cuando se fueron instalando paquetes sueltos
con `pip install <paquete>` en vez de instalar la lista completa de una vez. El error de `_lazywhere` de más
arriba y este de `xgboost` suelen tener la misma causa: un entorno incompleto o desactualizado.

**Solución.** Instale la lista completa, no el paquete señalado por el error uno por uno:

```
pip install -r requirements.txt
```

**Verificación**, para confirmar que los diez paquetes están presentes:

```
pip show streamlit pandas numpy plotly psutil scikit-learn scipy statsmodels xgboost TA-Lib
```

Si `pip install -r requirements.txt` se detiene en `TA-Lib`, revise la nota sobre esa dependencia en
`docs/despliegue_streamlit_cloud.md`: en Windows suele necesitar una rueda precompilada en lugar de compilarse
desde el código fuente. Si el problema persiste tras instalar la lista completa, lo más fiable es recrear el
entorno virtual desde cero (pasos en la sección anterior de este documento).

## AttributeError: Module 'scipy' has no attribute 'array' (al abrir Clustering jerárquico)

**Causa.** Esta sí era una falla de la aplicación, ya corregida, no de su entorno. La función
`plotly.figure_factory.create_dendrogram`, que se usaba para dibujar el dendrograma, llama internamente a
`scipy.array`. Esa función se retiró de `scipy` hace tiempo (es un alias antiguo de `numpy.array`). Versiones
de `plotly` anteriores a la que fija `requirements.txt` todavía dependen de ella; versiones más recientes ya
usan `numpy.array` y no fallan. Es un problema conocido y reportado en el propio repositorio de `plotly`
(issue #4495), sin corregir de forma consistente entre versiones.

**Solución.** La aplicación ya no depende de esa función interna de `plotly`. El dendrograma se calcula con
`scipy.cluster.hierarchy.dendrogram(..., no_plot=True)` (scipy solo calcula las coordenadas, no dibuja nada) y
se traza a mano con líneas de Plotly. Esto quita la dependencia frágil por completo, así que el error no
debería volver a aparecer con la versión actual del código, sin importar la versión exacta de `plotly`
instalada. Si aun así lo ve, actualice a la última versión del proyecto y confirme con:

```
pip show plotly
```

que tiene al menos la versión fijada en `requirements.txt`.

