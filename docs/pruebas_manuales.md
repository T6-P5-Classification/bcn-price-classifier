# Guía de comprobación por issue

Esta guía indica qué ejecutar y qué observar. Las cifras del snapshot son
referencias, no requisitos para cualquier dataset. Consultar la rama y los
archivos disponibles: una entrega pendiente de integración no está en `dev`
solo porque ya tenga una revisión o una propuesta abierta.

## Preparación y seguridad

1. Abrir la carpeta del proyecto en VS Code. Antes de cambiar de rama,
   comprobar Control de código fuente y archivos sin guardar. Si hay cambios,
   detenerse y consultar; no descartarlos ni ejecutar comandos de limpieza.
2. Para trabajo integrado, seleccionar `dev` actualizado. Para una entrega
   no integrada, usar su rama específica, solo con el directorio limpio.
3. Abrir el notebook desde `notebooks/` y seleccionar el kernel `.venv` del
   proyecto. Se necesita `data/raw/listings.csv.gz`, descrito en el README.
4. Reiniciar el kernel y ejecutar todas las celdas: Restart Kernel y Run All.
   Reiniciar borra las variables en memoria, no los archivos de datos.
5. Los notebooks indicados leen datos y calculan localmente; no escriben en
   servicios externos. Ejecutarlos modifica sus resultados en memoria;
   guardarlos cambia el archivo local. Para revisar no hace falta guardarlos.
   No publicar capturas o tablas que contengan información personal.
6. No aprobar, fusionar, configurar secretos ni ejecutar workflows como parte
   de esta revisión local. Cualquier prueba que cree registros, envíe datos,
   genere coste o no tenga una reversión clara requiere advertencia y
   autorización específica antes de ejecutarse.

## Issue 6 — Exploración inicial

- Archivo: `notebooks/01a_eda_initial.ipynb`, integrado en `dev`.
- Ejecutar todas las celdas y comprobar filas/columnas, tipos, nulos,
  histogramas, distribución de precios y tipos de alojamiento y boxplots.
- Esperado: ejecución sin errores, gráficos legibles y extremos visibles.
  Es una exploración histórica del raw, no la limpieza definitiva. No usar
  ahora el dataset completo para guiar decisiones reservadas a train.

## Issue 7 — Limpieza

- Archivo: `notebooks/01b_cleaning.ipynb`, integrado en `dev`.
- Ejecutar y comprobar filas eliminadas y precios nulos/no positivos finales.
- Esperado: cero precios nulos/no positivos en filas retenidas y conservación
  de extremos positivos. Para el snapshot actual: 12.610 filas limpias.
- Revisar nulos antes/después. `minimum_nights` puede seguir teniendo nulos;
  se imputan después, dentro del Pipeline, usando entrenamiento.

## Issue 8 — Selección de columnas

- Archivo: `notebooks/01c_column_selection.ipynb`, integrado en `dev`.
- Ejecutar y revisar columnas eliminadas y sus motivos.
- Esperado: coordenadas en modelo y mapa (`True` en ambas comprobaciones);
  precio fuera de las variables del modelo (`False` en la tercera).
- El precio sí puede estar en la vista del mapa. No exigir el mismo número
  de columnas antes y después de crear las variables derivadas.

## Issue 9 — Variables derivadas (dependencia)

- Archivo: `notebooks/01d_feature_engineering.ipynb`, integrado en `dev`.
- Ejecutar y comprobar `amenities_count`, `bathrooms_count`, antigüedad del
  anfitrión y variables de reseñas.
- Esperado: comprobaciones de columnas raw eliminadas y variables derivadas
  disponibles en el modelo con resultado `True`.
- Verificar `has_hair_dryer` (secador de pelo) y `has_clothes_dryer`
  (secadora de ropa). Su separación también tiene tests automatizados.

## Issue 12 — Split y categorías

- Archivo: `notebooks/01e_split_target.ipynb`, integrado en `dev`.
- Ejecutar y revisar tamaños, umbrales y proporciones de categorías.
- Referencias del snapshot: train 10.088 filas, test 2.522, 71 variables;
  límites mostrados de 122,00 y 249,72 euros por noche. No redondear los
  límites en el cálculo de etiquetas: la pantalla redondea, el código no.
- Esperado: tres categorías con aproximadamente un tercio de anuncios y
  aserciones finales sin errores. Los límites se aprenden en train y se
  reutilizan para test; los bins provisionales no son el target final.

## Issue 13 — Preprocesamiento y tests integrados

1. Abrir una terminal en la raíz y activar el entorno virtual. En macOS:

   ```bash
   source .venv/bin/activate
   python -m pytest -q
   ```

2. Esperado en la base integrada al redactar esta guía: 14 tests pasados.
   El número aumentará al incorporar nuevas entregas.
3. Los controles incluyen limpieza, target y preprocesador sin ajustar,
   nulos/categorías desconocidas, estadísticas de train y uso en Pipeline.
   No certifican la capacidad predictiva de un modelo.

## Issue 30 — Integridad de datos

Esta entrega está preparada en la rama `test/30-data-integrity`; comprobar
si ya fue integrada antes de elegir rama. Sus tests no requieren Airbnb.
La corrección se prepara con la base `dev` de commit `6795cd6`, manteniendo
el esquema de 23 variables acordado.

Desde la raíz, con el entorno virtual activado:

```bash
python -m pytest -q tests/test_data.py
python -m pytest -q
```

Esperado para esta entrega: 11 tests de integridad correctos. El total de
la suite depende de las otras entregas incorporadas; no usar el antiguo
total de 25 como contrato fijo.

Los controles comprueban precios válidos, separación e identidad de filas,
esquema y variables prohibidas, nulos permitidos, umbrales solo de train,
categorías coherentes, estratificación provisional, reproducibilidad,
precios exactamente en los límites y error con precios no separables.

1. Con la rama indicada y el entorno activado, ejecutar primero el comando
   específico y después la suite completa. Ambos realizan cálculos locales,
   crean archivos sintéticos temporales y pueden escribir cachés; no leen
   el CSV de Airbnb, no usan credenciales ni modifican servicios remotos.
2. En `tests/test_data.py`, comprobar que el caso de esquema exige todas las
   columnas de `MODEL_FEATURES` en orden y mantiene las prohibiciones
   independientes de precio e identificadores. La tabla ficticia debe aportar
   barrio, distrito y tipo de propiedad, además de las demás entradas.
3. En el caso de ausencias, verificar que se comparan los valores completos
   de `minimum_nights` con las filas originales, incluidos sus nulos, sin
   acceder a `minimum_nights_was_missing`.
4. Leer el README: deben estar presentes la guía de tests y la documentación
   vigente de modelos, sin marcadores de conflicto. Esta lectura no cambia
   archivos ni autoriza publicación. No se reentrenan modelos en esta corrección.
5. Si falla un test, anotar el nombre y el error y detener la publicación.
   No recuperar variables descartadas ni eliminar protecciones para forzar
   un resultado correcto. Un fallo de entorno se distingue de una regresión.

Resultado comprobado de esta corrección: antes, 2 fallos y 9 tests de
integridad correctos; después, 11 correctos y 56 correctos en la suite
completa con esta base. Además, se comprobó en memoria con datos sintéticos
que los controles rechazan una columna acordada ausente, columnas en orden
alterado, precio infiltrado y nulos de `minimum_nights` rellenados antes de
tiempo. Estas comprobaciones no modificaron código de producción ni datos reales.

## Issue 10 — EDA por categoría real

Rama de implementación: `feat/10-eda-classes` hasta que esté integrada.
Archivo: `notebooks/02_eda_classes.ipynb`.

1. Con datos y kernel preparados, reiniciar el kernel y ejecutar todo.
2. En la primera salida, comprobar que solo se analiza entrenamiento:
   referencia de 10.088 anuncios y 71 variables para el snapshot actual.
   Test se descarta tras la carga común; no se grafica ni se calculan sus
   distribuciones o métricas.
3. Revisar tabla y gráfico de categorías **reales según precio/umbrales**.
   Referencias de train: Económico 3.366, Medio 3.359 y Premium 3.363.
   No son predicciones ni evidencia de tres segmentos naturales iguales.
4. Revisar las medianas, observaciones disponibles y porcentajes ausentes de
   las 12 variables seleccionadas. Los porcentajes deben estar entre 0 y 100;
   no exigir cero nulos ni interpretar medianas sin revisar su cobertura.
5. Revisar los ocho histogramas: mismos colores/orden de categorías, leyenda
   «Categoría real», proporción dentro de cada categoría e intervalos comunes.
   Camas, estancia mínima y reseñas usan `log(1 + valor)` solo para visualización;
   no cambia el entrenamiento ni elimina extremos. Revisar la tabla de rangos.
6. Revisar la matriz Spearman de características: escala de −1 a +1, sin
   precio ni códigos numéricos del target. La tabla de pares incluye tamaños
   de muestra y coeficientes globales y por categoría real. Correlación no
   significa causalidad ni importancia del modelo.
7. Leer las observaciones calculadas y límites. Para el snapshot actual,
   capacidad mediana 2/4/5 personas y amenities medianos 24/30/34, en orden
   Económico/Medio/Premium. Son referencias que pueden cambiar con los datos.
8. Esperado al final: mensaje de comprobaciones correctas, sin modificar las
   variables originales. No hay entrenamiento, mapas ni selección automática.
9. Ejecutar `python -m pytest -q` desde la raíz como comprobación adicional
   del código existente. El éxito del notebook y el de los tests son hechos
   separados; no se han evaluado métricas predictivas en esta issue.

## Issue 20 — XGBoost con 40 pruebas de Optuna

1. **Preparación:** desde la raíz del proyecto, usar `feat/20-xgboost` hasta
   que se integre, entorno `.venv` y el CSV del README. Comprobar
   `python -c "import xgboost; print(xgboost.__version__)"`. Solo lee software
   instalado; si falla por OpenMP, no entrenar hasta resolver la dependencia
   siguiendo el README. No reinstalar dependencias compartidas sin revisar.
2. **Tests sintéticos:** ejecutar `python -m pytest -q tests/test_xgboost_model.py`
   y después `python -m pytest -q`. Ejecuta ajustes pequeños con datos sintéticos;
   no lee el CSV real ni evalúa calidad predictiva en producción. Puede crear
   cachés locales. Esperado: todas las pruebas pasan.
3. **Inspección de entrega:** abrir `notebooks/06_model_xgboost.ipynb` y los
   archivos `docs/results/xgboost_*`. Esta lectura no cambia registros.
   Comprobar 23 variables, 40 pruebas completas, cinco folds por prueba,
   candidato con gap < 0,05 y test sin evaluar. La tabla de los diez mejores
   trials no es el registro completo: el CSV de trials debe tener 40 filas.
4. **Reproducción opcional:** antes de repetir, conservar copia del notebook
   y de los tres archivos agregados si se quiere mantener la ejecución anterior.
   Ejecutar `python tools/run_xgboost_notebook.py` desde la raíz. Lee el CSV,
   consume CPU y reemplaza esas cuatro salidas; no usa servicios remotos, no
   modifica datos y no guarda modelos ni predicciones individuales. No ejecutar
   a la vez desde el editor. Esperado: ejecución completa sin errores de celda.
5. **Métricas:** en `xgboost_details.json`, comprobar `test_evaluated=false`,
   `nested_cv=false`, 23 campos en `input_schema`, mapeo Económico=0/Medio=1/
   Premium=2 y registro de versiones/firmas. Revisar `selected_cv`: media train
   menos media validación debe coincidir con gap, sin redondear para aprobarlo.
   No confundir la media de F1 por fold con el F1 agregado OOF.
6. **Errores:** cada matriz debe sumar las filas de entrenamiento (10.088 para
   este snapshot); filas reales y columnas predichas, orden común de clases.
   Revisar precision/recall/F1 por categoría. La tasa extrema divide ambas
   confusiones Económico/Premium entre TODOS los anuncios OOF.
7. **Integración:** el CSV de comparación tiene las 18 columnas descritas en
   `docs/xgboost.md` y no tiene columnas `test_*`. Confirmar que el lector común
   ya está integrado antes de exigir que aparezca XGBoost en la comparación.
   Esta issue no sustituye ni implementa ese lector o la elección del champion.
8. **Fallo:** detenerse ante columna ausente, error de celda o ningún trial
   elegible. Anotar celda/comando y mensaje, comprobar datos/entorno y solicitar
   revisión; no borrar tests, ampliar pruebas o consultar test para forzar
   aceptación. Un archivo de resultados previo no acredita una ejecución fallida.

Estas comprobaciones no autorizan publicación, aprobación ni merge. Los
resultados concretos de la ejecución se resumen en `docs/xgboost.md`.

## Revisión de entregas documentales y automatización

- Decisiones comunes: leer `docs/decisiones.md` en la versión correspondiente
  a la entrega de la issue 14. Distinguir acuerdos, evidencias y propuestas.
- Directrices: si `AGENTS.md` está disponible, revisar sus reglas con el equipo.
  Leer una propuesta no equivale a aprobarla o fusionarla.
- Automatización: leer su guía específica cuando esté disponible; no pulsar
  Run workflow antes de que Esra confirme permisos, credencial, caso de prueba
  y autorización. Una ejecución puede mover tarjetas y cerrar issues, dejando
  historial persistente. Los tests simulados no prueban esos permisos reales.

## Qué registrar y cómo tratar un fallo

```text
Issue o entrega:
Rama o commit revisado:
Archivo/pantalla/comando:
Resultado esperado:
Resultado observado:
Errores o dudas:
```

- `passed` indica que se cumple el caso automatizado definido. `FAILED`
  requiere revisar el nombre y la comparación; no borrar el test para ocultarlo.
- Si falla un notebook, anotar la celda y el error sin publicar datos
  personales. Un archivo ausente, kernel incorrecto o dependencia faltante no
  demuestran por sí solos un fallo en el análisis.
- Que exista una entrega, pasen tests, esté aprobada y esté fusionada son
  estados distintos. No sustituir uno por otro.
- Mantener aquí los pasos de comprobación de cada nueva issue; los acuerdos
  y motivos pertenecen a `docs/decisiones.md`. No duplicar conversaciones.
