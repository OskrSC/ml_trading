# Despliegue en Streamlit Community Cloud

## Antes de empezar

- El archivo principal es `streamlit_app.py`, en la raíz del repositorio.
- El servicio lee un solo archivo de dependencias. Aquí es `requirements.txt`, en la raíz. Los paquetes del sistema (apt) irían en `packages.txt`, que hoy no se necesita.
- Conviene fijar la versión de Streamlit, como ya hace `requirements.txt`, para que el servicio no la actualice por su cuenta.
- Desarrolle y despliegue con la misma versión de Python. El proyecto se probó con Python 3.12.

## Pasos

1. Suba el contenido de esta carpeta a un repositorio de GitHub.
2. En share.streamlit.io, inicie sesión y cree una app nueva. Elija el repositorio, la rama y el archivo principal `streamlit_app.py`.
3. Abra "Advanced settings" y elija Python 3.12. Ahí también se pueden escribir secretos.
4. Despliegue y revise los registros de compilación. Los puntos de riesgo son TA-Lib, xgboost y statsmodels.

Cada cambio que haga en GitHub se refleja en la app. Si toca `requirements.txt`, el servicio reinstala las dependencias.

## Si TA-Lib no se instala

1. Comente la línea `TA-Lib` de `requirements.txt` y vuelva a desplegar. La app arranca igual y la página de diagnóstico muestra el paquete como no instalado.
2. Pruebe con `packages.txt` si necesita librerías del sistema para compilarlo.
3. Registre el resultado en `docs/informe_fase0.md`. La fase 1 incluye un cálculo de indicadores sin TA-Lib como camino de respaldo.

## Límites del servicio

La documentación de Community Cloud publica, con fecha de febrero de 2024, unos límites aproximados de 0,078 a 2 núcleos de CPU, de 690 MB a 2,7 GB de memoria y hasta 50 GB de disco. Pueden cambiar sin aviso. La página de diagnóstico muestra el límite real del contenedor cuando el sistema lo expone. Si la app supera los límites, se ralentiza o deja de responder.

Las apps sin uso se ponen en reposo y la primera visita posterior arranca en frío. Consulte la documentación oficial para el plazo vigente.

## Medir la fase 0

1. Abra la app desplegada y entre en Diagnóstico del entorno.
2. Pulse "Actualizar mediciones", "Medir la carga de datos" y "Medir importación".
3. Descargue el informe JSON.
4. Vuelva a las mismas mediciones tras una hora sin uso, para registrar el arranque en frío.
5. Copie los datos a `docs/informe_fase0.md`.

## Experimento con TensorFlow (opcional)

El perfil elegido es ligero y no lo necesita. Solo hágalo si desea reconsiderar el perfil completo.

Las versiones fijadas en `requirements.txt` admiten `tensorflow==2.21.0` (con Keras 3) en Python 3.12, según una simulación de instalación. No se midió en la nube.

1. Añada temporalmente `tensorflow==2.21.0` a `requirements.txt` y despliegue.
2. Anote si el build termina y cuánto tarda.
3. En Diagnóstico, mida la importación de `tensorflow`. Si el servicio termina el subproceso, la tabla lo indicará.
4. Con la app en reposo, compare la memoria del contenedor contra su límite.
5. Revierta el cambio si el resultado no cabe con margen. Ver el criterio en `docs/informe_fase0.md`.

## Ocultar el diagnóstico antes del lanzamiento

Añada en los secretos del servicio:

```
MLT_MOSTRAR_DIAGNOSTICO = "0"
```

Los secretos de nivel superior también quedan disponibles como variables de entorno, y la app lee ambos.

## Entornos de prueba y producción

Streamlit Community Cloud liga una app a un repositorio y una rama concretos: no ofrece por sí mismo el
concepto de "entorno de prueba" y "entorno de producción" separados. El patrón habitual para tenerlos es
desplegar dos apps distintas a partir del mismo repositorio:

- Una app de producción, apuntando a la rama `main` (o la que se use como estable).
- Una app de prueba, apuntando a una rama `staging` o `dev`, con su propia URL y sus propios secretos
  (puede compartir el mismo `MLT_PERFIL`, pero conviene que `MLT_MOSTRAR_DIAGNOSTICO` quede en `1` ahí, para
  poder revisar recursos y registros antes de fusionar a producción).

Fusionar cambios a la rama de producción, tras probarlos en la app de prueba, sirve como mecanismo de
reversión implícito: si algo falla, se revierte el commit y Streamlit Cloud vuelve a desplegar la versión
anterior. Esta aplicación no crea esas dos apps por adelantado, porque implica una decisión de costo y de
infraestructura que corresponde a quien la despliegue.

## Al lanzar

`.streamlit/config.toml` ya trae `showErrorDetails = "type"` por defecto desde la fase 6: en producción se ve
el tipo de error, no la traza completa ni rutas de archivo del servidor. Se genera desde
`mltrading/config/theme_config.py`; si alguien lo cambió a `"full"` para depurar en local, confirme que se
revirtió antes de desplegar (`python scripts/generate_theme.py` lo regenera con el valor correcto, y
`tests/test_tokens.py` falla si el archivo versionado no coincide con lo que generaría ese script).

Repase también `docs/checklist_lanzamiento.md`, que reúne en un solo lugar el resto de comprobaciones antes
de publicar la aplicación.
