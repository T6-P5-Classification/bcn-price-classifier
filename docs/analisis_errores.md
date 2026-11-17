# Análisis de errores: categorías vecinas y extremas (#17)

## Alcance y coordinación

Rama de entrega: `feat/17-neighbour-class-errors`, desde `dev` en `e2e13e1`.
Notebook: `notebooks/08_comparison.ipynb`, sección «Análisis de errores entre
categorías vecinas — issue 17». Funciones verificables: `src/error_analysis.py`.

**Discrepancia de Random Forest: se conserva y se presenta por separado.**
El primer intento se detuvo antes de exportar. Tras revisar el origen, el
usuario autorizó continuar distinguiendo métricas originales y diagnóstico
local, sin modificar los CSV de modelos ni ocultar diferencias.

Con el mismo CSV, código de preparación, código de Random Forest, parámetros
y particiones, la accuracy OOF local de RF es 0,7595162569389373 frente a
0,7588223632038065 del CSV: 7.662 frente a 7.655 aciertos entre 10.088 anuncios.
Se contrastaron los tres archivos de origen con el commit declarado por RF,
`9bb15b1`: son idénticos. Además, ejecutando directamente `oof_diagnostics()`
y `cv_metrics()` de `src.rf_model`, sin las funciones nuevas, aparece la
misma discrepancia. Accuracy OOF local 0,7595163; F1 macro OOF 0,7600899;
F1 medio CV 0,7600723 y gap medio 0,0452971. La tasa extrema coincide con
origen: 98/10.088. La selección de RF es coherente con sus 40 pruebas:
ocho elegibles y candidato de índice 8.

La causa aún no está verificada. El notebook de origen declara Python 3.14.7
y el entorno local usa 3.15.0; no se demuestra que eso explique la diferencia.
No se atribuye automáticamente al entorno ni a un error del autor del modelo.
Conviene que el responsable confirme las versiones efectivas de las librerías
y conserve matriz e informe originales; no se modifica su entrega desde #17.

El contraste estricto conserva tolerancia absoluta 1e-9. Solo RF tiene una
excepción explícita para continuar con diferencias visibles: el JSON conserva
valores de origen, locales, diferencia y booleano de coincidencia por métrica.
No se declara una reproducción exacta cuando no existe. El lector original
y sus cifras de selección permanecen intactos. Los demás modelos mantienen
la detención ante diferencias no acordadas.

Se añade una sección separada después del lector de CSV existente, sin
sustituirlo, cambiar su ordenación ni declarar un champion. Se analizan las
cuatro configuraciones seleccionadas y ya entregadas: regresión logística,
Random Forest, XGBoost y LightGBM. No se comparan los trials de sus búsquedas
ni las referencias Dummy duplicadas.

La elección del champion y su evaluación final pertenecen a la entrega
compartida de selección. Este trabajo aporta evidencia, no decide por su
responsable. No incluye análisis por distrito, modalidad, cercanía a los
límites, costes monetarios ni cambios de parámetros, variables o aplicación.

## Qué significa cada tipo de error

La categoría **real** se construye desde el precio anunciado por noche y los
umbrales comunes. La categoría **predicha** es la salida del modelo. No se
clasifica lujo ni calidad. El orden es Económico, Medio, Premium.

| Categoría REAL | Categoría PREDICHA | Tipo |
|---|---|---|
| Económico | Medio | Vecino |
| Medio | Económico | Vecino |
| Medio | Premium | Vecino |
| Premium | Medio | Vecino |
| Económico | Premium | Extremo |
| Premium | Económico | Extremo |

Una coincidencia entre real y predicha es un acierto. Vecino/extremo describe
separación entre categorías ordenadas: no mide euros ni asigna costes. Un
error vecino no tiene necesariamente la misma consecuencia que otro vecino.

Las matrices tienen filas reales y columnas predichas, siempre en el orden
común. Cada figura muestra recuentos a la izquierda y porcentajes dentro de
la categoría real a la derecha. Las filas observadas de la segunda matriz
suman 100 %. No son porcentajes dentro de la categoría predicha.

## Denominadores que no deben mezclarse

- `n_total`: todos los anuncios de train evaluados fuera de fold.
- `n_correct`: predicciones que coinciden con la categoría real.
- `n_errors`: todas las predicciones equivocadas; vecinos más extremos.
- `n_neighbour_errors` y `n_extreme_errors`: recuentos de esos tipos.
- `pct_*_of_all`: porcentaje sobre **todos los anuncios**, incluidos aciertos.
  El gráfico de errores omite los aciertos y sus barras no suman 100 %.
- `pct_*_of_errors`: porcentaje sobre **las predicciones equivocadas** del
  modelo. Vecinos y extremos suman 100 % si existen errores. Un porcentaje
  alto de vecinos no prueba buen rendimiento sin mirar cuántos errores hay.
- En la tabla de seis direcciones, `pct_of_real_class` usa solo anuncios de
  la categoría real de esa fila; `n_real_class` muestra ese denominador.

Si un modelo no comete errores, los porcentajes sobre errores no están
definidos: se conservan ausentes y el gráfico dice «Sin errores», no 0 %.
Una categoría real sin ejemplos también conserva ausentes sus porcentajes
por fila. El JSON expresa esos casos como `null`, no como ceros inventados.

## Reconstrucción de predicciones, sin nueva búsqueda

Los CSV de comparación guardan métricas agregadas, no predicciones por
anuncio. Se repiten **cinco ajustes por modelo, 20 en total**, con parámetros
ya fijados, para obtener una predicción fuera del ajuste por fila de train.
No se ejecuta Optuna ni se ajusta un modelo final con todo train.

1. El lector existente comprueba los CSV y los presenta por F1 macro medio
   de validación. Se conserva su resultado intacto.
2. Se exige exactamente una configuración de cada uno de los cuatro modelos,
   protocolo común, diagnósticos `train_oof`, snapshot/hash compatibles y gap
   medio < 0,05. No se ignoran candidatos ausentes o duplicados en silencio.
3. Se utiliza `load_train_test(return_thresholds=True)`. Esta carga crea los
   objetos reservados, pero se descartan sin inspección ni puntuación.
4. Se comprueba que la firma del CSV local coincida con la de los candidatos
   y se mantienen las 23 entradas y los umbrales sin redondear.
5. `StratifiedKFold(5, shuffle=True, random_state=42)` se materializa una vez;
   `cross_val_predict` recibe cada Pipeline completo y esos mismos repartos.
   Imputación, escalas y categorías se aprenden dentro de cada ajuste.
6. Se leen los parámetros de cada fila. La regresión logística conserva
   además `solver=lbfgs` y `max_iter=5000` de su notebook de origen. Random
   Forest y XGBoost mantienen Económico=0, Medio=1, Premium=2 y se traducen
   a nombres para interpretar resultados. Los otros modelos conservan texto.
7. Se contrastan accuracy, F1 macro OOF y tasa extrema frente a los CSV, con
   tolerancia absoluta 1e-9. Una discrepancia detiene el análisis salvo la
   excepción de RF autorizada y documentada arriba. En ese caso se muestran
   ambas cifras sin cambiar el resultado de origen, datos, modelos o test.

Se limita el proceso a un hilo y ajustes secuenciales. La reconstrucción no
constituye otra búsqueda de mejores configuraciones y no aumenta su presupuesto.
Los 20 ajustes describen cada ejecución completa, no todo el trabajo de
verificación: se repitieron ejecuciones idénticas para comprobar la salida y
su presentación. El primer intento detenido y la comprobación por la ruta
original de RF consumieron también ajustes fijos, sin nuevos trials. Los
tests sintéticos son adicionales.

## Archivos y efectos de ejecución

- `docs/results/error17_summary.csv`: cuatro filas de recuentos y porcentajes.
- `docs/results/error17_directions.csv`: seis direcciones por modelo, 24 filas.
- `docs/results/error17_details.json`: matrices, porcentajes por fila,
  configuración efectiva, filas de origen, firmas, versiones, tiempos y límites.
- Figuras y tablas incrustadas en el notebook: cuatro figuras de matrices
  dobles y dos comparaciones con denominadores distintos.

No se guardan anuncios, predicciones individuales, precios, identificadores
ni Pipelines ajustados. Los archivos `error17_*` no terminan en
`_comparison.csv`, de modo que el lector de modelos no los confunde con sus
entradas. El commit registrado identifica la base; las firmas de código
identifican también funciones locales aún no publicadas.

Requisitos: entorno del proyecto, herramientas de `requirements-notebooks.txt`,
CSV local del README y las cuatro salidas `*_comparison.csv` integradas.
No se necesita GeoJSON, MongoDB, Docker ni claves de servicios. En macOS,
XGBoost/LightGBM necesitan OpenMP, ya descrito en el README.

Desde la raíz:

```sh
source .venv/bin/activate
python -m pytest -q tests/test_error_analysis.py tests/test_error_analysis_results.py tests/test_comparison_notebook.py
python tools/run_error_analysis_notebook.py
```

Los tests usan datos sintéticos, matrices y agregados guardados; pueden crear
cachés locales. No necesitan el CSV de Airbnb. El último comando sí lee el
CSV real, consume CPU y realiza los 20 ajustes fijados. Reemplaza el notebook
y los tres archivos `error17_*`: conservar una copia antes si se desea
mantener una ejecución anterior. No modifica los CSV de modelos ni raw,
no publica nada y no accede a servicios externos.

No usar el ejecutor genérico sin revisar sus argumentos: su opción por
defecto también ejecuta la baseline de regresión logística, que incluye su
propia búsqueda. El ejecutor específico anterior ejecuta solamente este
notebook. Tampoco ejecutar a la vez desde editor y terminal.

Los tests del lector se limitan a sus celdas originales. Las celdas nuevas
están etiquetadas `issue17` y tienen tests separados, para no convertir las
pruebas de lectura de CSV sintéticos en entrenamiento de cuatro modelos.
Las validaciones anteriores del lector se conservan.

## Lo que las conclusiones no demuestran

- **No es evaluación final independiente:** los parámetros y las variables
  se escogieron consultando esta misma validación. OOF evita que una fila
  participe en el ajuste que la predice, pero no deshace esa selección previa.
- La definición de clases usa umbrales de train completo y el reparto usa
  estratificación provisional global. No se evalúa un protocolo distinto.
- El reparto no separa anfitriones ni tiempo: no demuestra rendimiento en
  anfitriones desconocidos o anuncios futuros.
- El proyecto expuso históricamente resultados del test de la baseline.
  Este análisis no consulta test, pero no convierte esa partición en inédita.
- Una diferencia pequeña de errores no demuestra superioridad estable ni
  significación. El F1 agregado OOF no sustituye al F1 medio por fold de la
  selección. No se cambia la ordenación del lector para favorecer un modelo.
- Las matrices agregadas no permiten afirmar que dos modelos fallan en los
  mismos anuncios, ni explicar por qué. Eso exigiría otro análisis acordado.
- No se atribuyen causas a personas, distritos o modalidades ni se inventan
  costes económicos. Champion e integración permanecen fuera del alcance.

Los pasos de aceptación están en la sección de la issue 17 de
`docs/pruebas_manuales.md`. No se cambian estados del Project ni se autoriza
publicación por ejecutar verificaciones locales.

## Resultados comprobados de la reconstrucción local

Población común: **10.088 anuncios de entrenamiento**, 23 entradas. La tabla
original de selección no se modifica. Las cifras siguientes son del análisis
OOF local: RF difiere de su origen y se identifica como tal en CSV, JSON y
notebook. Los otros tres modelos reproducen sus métricas OOF guardadas.

| Modelo seleccionado | Errores totales | Vecinos | Extremos | Vecinos / todos los anuncios | Extremos / todos los anuncios | Vecinos / errores |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | 2.727 | 2.601 | 126 | 25,783 % | 1,249 % | 95,380 % |
| Random Forest, local distinto del origen | 2.426 | 2.328 | 98 | 23,077 % | 0,971 % | 95,960 % |
| XGBoost | 2.391 | 2.289 | 102 | 22,690 % | 1,011 % | 95,734 % |
| LightGBM | 2.393 | 2.295 | 98 | 22,750 % | 0,971 % | 95,905 % |

Lecturas que permiten estas cifras:

- Entre el 95,38 % y el 95,96 % de los errores observados son vecinos, pero
  eso no convierte a los modelos en infalibles: siguen equivocándose en
  aproximadamente el 23,70–27,03 % de todos los anuncios evaluados.
- La dirección más frecuente en los cuatro modelos es **categoría real
  Premium, categoría predicha Medio**: 913 casos en regresión logística,
  821 en RF local, 832 en XGBoost y 829 en LightGBM. No prueba la causa ni
  indica si esos precios estaban cerca del límite entre categorías.
- XGBoost tiene dos aciertos más que LightGBM en esta reconstrucción, pero
  también cuatro errores extremos más. Eso muestra por qué una única cifra
  no describe todos los fallos. No demuestra superioridad estable ni asigna
  un ganador sin costes de errores acordados y evaluación final.
- La diferencia de RF es de siete aciertos netos, no evidencia de que solo
  cambiaran siete predicciones: no se conservan las predicciones individuales
  de origen para hacer esa comparación por anuncio.

## Verificación realizada y límites de la revisión de RF

- 40 trials de RF completos, ocho elegibles y candidato de índice 8: selección
  y parámetros coherentes entre el registro de pruebas y el CSV original.
- Los 11 tests existentes de RF pasan. No prueban coincidencia de métricas
  reales entre entornos. No se repitieron sus 40 pruebas de Optuna.
- Ruta original de RF ejecutada con parámetros fijos: cinco ajustes OOF y
  cinco de CV; confirma que la diferencia también existe sin código de #17.
- Notebook de comparación/análisis completo: siete celdas de código, sin
  errores, cuatro matrices dobles y dos gráficos con denominadores distintos.
- Tests propios: 26 correctos. Suite completa: **110 correctos** con esta
  base. Incluye pruebas sintéticas y coherencia de agregados guardados; no
  implica calidad demostrada en producción ni causa verificada de la diferencia.
