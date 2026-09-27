# Lista de verificación antes del lanzamiento

Consolida los pasos de despliegue y endurecimiento repartidos por `docs/`. Márquelos al completarlos.

## Antes de desplegar

- [ ] `python -m pytest` en verde en un entorno limpio (`pip install -r requirements-dev.txt`, sin caché).
- [ ] `pip-audit -r requirements.txt` sin vulnerabilidades de gravedad alta o crítica sin revisar.
- [ ] Confirmar o sustituir la licencia sugerida del código propio (`NOTICE`, `LICENSE`; ver
      docs/mantenimiento.md).
- [ ] `.streamlit/config.toml` tiene `showErrorDetails = "type"` (ya es el valor por defecto desde la fase 6;
      compruebe que nadie lo cambió a `"full"` para depurar y olvidó revertirlo).

## Desplegar en streamlit.app

- [ ] Seguir `docs/despliegue_streamlit_cloud.md`: repositorio, Python 3.12, archivo principal
      `streamlit_app.py`.
- [ ] Build sin errores, con TA-Lib, xgboost y statsmodels instalados (o el plan de respaldo de TA-Lib, si
      hiciera falta).
- [ ] Definir el secreto `MLT_MOSTRAR_DIAGNOSTICO = "0"` para ocultar la página de diagnóstico al público.
- [ ] Confirmar el perfil de despliegue (`MLT_PERFIL`, ligero por defecto) y que coincide con lo decidido.

## Comprobaciones manuales en la app ya desplegada

- [ ] El menú superior derecho muestra Sistema, Claro y Oscuro, y el cambio es instantáneo.
- [ ] Las fuentes Source Sans 3 y Newsreader cargan (fase 0: no se pudo comprobar sin acceso a internet en el
      entorno de desarrollo).
- [ ] Las 15 páginas cargan sin error, en escritorio y en móvil.
- [ ] Un flujo completo funciona de principio a fin: Variables y objetivo → División de datos → Modelos
      supervisados (entrenar) → Evaluación → Backtesting, con la marca "Resultado predeterminado" visible.
- [ ] Descargar y volver a cargar una receta reproduce el mismo resultado.
- [ ] El simulador de Operación en vivo avanza sin error con su configuración predeterminada.
- [ ] Subir un archivo de receta que no sea JSON, o mayor de 64 KB, produce un mensaje de error claro y no
      tumba la página.

## Rendimiento y recursos

- [ ] Página Diagnóstico del entorno: memoria del contenedor y su límite, con margen razonable en reposo.
- [ ] Tiempo de arranque en frío (primera visita tras un periodo de reposo) medido y anotado en
      `docs/informe_fase0.md`.
- [ ] Completar las secciones pendientes de `docs/informe_fase0.md` con las mediciones reales del servicio.

## Después del lanzamiento

- [ ] Revisar la salida estándar del servicio (o la página de diagnóstico) tras las primeras horas de uso, en
      busca de errores no previstos.
- [ ] Anotar en `docs/decisiones.md` cualquier ajuste hecho ya en producción que no estuviera documentado
      antes del lanzamiento.
