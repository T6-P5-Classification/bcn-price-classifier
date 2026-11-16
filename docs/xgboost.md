# XGBoost: entrenamiento y diagnósticos de la issue 20

## Alcance

Rama de entrega: `feat/20-xgboost`, creada desde `dev` en `6cf7cb7`.
Notebook: `notebooks/06_model_xgboost.ipynb`.
Funciones verificables: `src/xgboost_model.py`.

Se usa el EDA integrado del equipo y sus **23 variables**. No se incorpora
otro EDA, no se cambia el formulario ni se implementa el análisis adicional
de errores o la elección del champion. La predicción es un segmento de
precio anunciado por noche, no una medida de lujo ni un precio recomendado.

## Presupuesto y criterio de selección

Se acordaron **40 pruebas de Optuna para XGBoost**, con cinco particiones
por prueba: 200 ajustes de búsqueda. Este presupuesto coincide con el
utilizado en LightGBM, pero no modifica ni repite las búsquedas de los demás
algoritmos; un presupuesto uniforme del equipo requiere su propio acuerdo.

La búsqueda TPE utiliza semilla 42, dirección de maximización del F1 macro
medio de validación, un proceso y un hilo. No tiene poda, parada temprana,
timeout de búsqueda ni ampliación automática del presupuesto. Además se
realizan diez ajustes de referencia (Dummy y XGBoost sin búsqueda) y diez
para diagnósticos OOF (Dummy y configuración elegida): 220 ajustes en total.

Incidencia del lanzador: un primer intento fue interrumpido por el límite
externo de dos minutos antes de guardar resultados. No se conoce cuántos
ajustes parciales alcanzó a ejecutar. Se reinició desde cero con la misma
semilla y espacio, sin reutilizar puntuaciones del intento interrumpido.
Los 220 ajustes describen la ejecución completa; el coste total de cómputo
incluye también trabajo duplicado de aquel intento, no cuantificado.

Las 40 pruebas exploran estos parámetros, fijados antes de ejecutarlas:

| Parámetro | Rango | Qué controla |
|---|---|---|
| `n_estimators` | 100–500, saltos de 50 | Número de árboles |
| `max_depth` | 2–6 | Profundidad máxima de cada árbol |
| `learning_rate` | 0,02–0,15, escala logarítmica | Peso de cada nueva etapa |
| `min_child_weight` | 5–100, escala logarítmica | Peso mínimo necesario en una hoja |
| `gamma` | 0–5 | Ganancia mínima para dividir |
| `subsample` | 0,6–1 | Proporción de filas utilizada por árbol |
| `colsample_bytree` | 0,6–1 | Proporción de columnas utilizada por árbol |
| `reg_alpha` | 0,001–10, escala logarítmica | Penalización L1 de los pesos |
| `reg_lambda` | 0,1–100, escala logarítmica | Penalización L2 de los pesos |

Se mantienen `tree_method='hist'`, `objective='multi:softprob'`, tres clases,
`eval_metric='mlogloss'`, semilla 42 y `n_jobs=1`. Aunque el estimador define
mlogloss como su métrica interna, **Optuna se selecciona por F1 macro**, no
por logloss. No se suministra un conjunto de test para ajustar o detener árboles.

Al terminar, se elige la prueba completa con mayor F1 de validación entre
las que cumplen `media_F1_train - media_F1_validación < 0.05`. Una igualdad
exacta de 0,05 no cumple. Si hay empate exacto de F1, se elige la prueba más
temprana. Se informa también la mejor prueba sin esa restricción y cuántas
son elegibles. Si ninguna cumple, se conserva el CSV de pruebas y el notebook
se detiene con un error explícito: no se aumenta la búsqueda ni se declara
éxito. Los gaps negativos se informan sin recortarlos.

## Datos, validación y test reservado

- `load_train_test(return_thresholds=True)` y `build_preprocessor()` son las
  únicas rutas de preparación. No se duplican limpieza, selección o reparto.
- La carga crea ambos objetos de test, pero se eliminan inmediatamente:
  no se muestran sus valores, distribuciones, predicciones ni puntuaciones.
- `StratifiedKFold(5, shuffle=True, random_state=42)`, materializado una vez.
- Cada Pipeline aprende imputación, escalas y categorías dentro de su fold.
- Mapeo explícito: Económico=0, Medio=1, Premium=2. No se usa orden alfabético.
- F1 macro de validación: media entre cinco folds. Desviación poblacional,
  `ddof=0`, para respetar el formato previsto para la tabla común.
- `cross_val_predict` recibe el Pipeline completo y los mismos folds. Sus
  predicciones sobre train producen accuracy, informe por clase, matrices y
  tasa de errores extremos; no se guardan predicciones individuales.
- Filas de matriz: categoría **real** según precio/umbrales. Columnas:
  categoría **predicha**. Orden Económico, Medio, Premium en ambas.
- Errores extremos: real Económico/predicho Premium y real Premium/predicho
  Económico, divididos por todos los anuncios evaluados, no solo los extremos.

Los umbrales finales se calculan con train completo antes de los folds y el
reparto usa grupos provisionales calculados sobre todo el archivo limpio.
La selección de variables también consultó esta validación. Optimizar y
diagnosticar con los mismos folds no es validación anidada ni una evaluación
final independiente. El reparto aleatorio no separa anfitriones ni tiempo.
La baseline del proyecto expuso históricamente resultados de la partición
reservada: este notebook no lo repite, pero no borra ese antecedente.

## Salidas e integración

Solo se generan:

- `docs/results/xgboost_comparison.csv`: candidato XGBoost y referencia Dummy,
  con las 18 columnas previstas para el lector común de comparación.
- `docs/results/xgboost_trials.csv`: 40 filas, una por prueba, estado, parámetros,
  tiempos y métricas agregadas.
- `docs/results/xgboost_details.json`: metadatos, versiones, firmas del CSV y
  del código ejecutado, esquema de entrada, umbrales sin redondear, mapeo,
  resultados por fold, matrices e informes agregados.
- Resultados y gráficos incrustados en el notebook.

El lector multiarchivo de comparación aún no pertenece a la base de esta
rama. Al integrarlo, podrá descubrir `xgboost_comparison.csv` sin que esta
issue modifique `08_comparison.ipynb`. La compatibilidad de columnas no
demuestra por sí sola comparabilidad: verificar snapshot, hash, protocolo,
versión de datos/código y presupuestos. No se publica un champion ni se
exporta un modelo ajustado con todo train.

## Ejecución y efectos locales

Con la rama indicada y el CSV descrito en el README, desde la raíz:

```sh
source .venv/bin/activate
python -c "import xgboost; print(xgboost.__version__)"
python -m pytest -q
python tools/run_xgboost_notebook.py
```

El test automatizado usa datos sintéticos y no representa rendimiento real.
El último comando lee el CSV real, consume CPU, ejecuta una búsqueda nueva de
40 pruebas y reemplaza el notebook y los tres archivos agregados de XGBoost.
Antes de repetirlo conserva una copia de esos cuatro archivos si quieres
mantener una ejecución anterior. No modifica los datos, no publica nada, no
escribe en una base de datos ni requiere servicios con coste.

Herramientas adicionales del ejecutor: `requirements-xgboost-notebook.txt`.
Instalarlas modifica el entorno Python local; usa un entorno separado si
necesitas preservar el existente. En macOS el README requiere `libomp`; su
instalación se autorizó y se verificó antes de ejecutar esta entrega. No se
modifican rutas personales en el código ni el entorno de otros compañeros.

El notebook puede ejecutarse también desde el editor con el kernel del
proyecto, tanto desde la raíz como desde `notebooks/`. No ejecutar ambas
opciones a la vez. El ejecutor guarda evidencia de la celda fallida si hay
un error; ese error no convierte archivos de una ejecución anterior en
resultados de la nueva. Comprueba siempre ejecución completa y metadatos.

Consulta la sección de la issue 20 en `docs/pruebas_manuales.md` para aceptar
la entrega. El éxito local no autoriza commit, push, apertura o merge de PR.

## Resultados comprobados de esta ejecución

Datos reales: snapshot Barcelona del 17/09/2026, 10.088 anuncios de train,
23 entradas. Python 3.15.0, XGBoost 3.4.1, scikit-learn 1.9.1 y Optuna 5.0.0.
El README propone Python 3.12; esta ejecución se verificó en el entorno local
3.15 y no demuestra por sí sola la reproducción en 3.12 u otros sistemas.
El hash del archivo y las versiones completas están en `xgboost_details.json`.

| Configuración | F1 macro medio ajuste | F1 macro medio validación | Desviación de validación | Gap |
|---|---:|---:|---:|---:|
| Dummy mayoritario | 0,16679 | 0,16679 | 0,00007 | ≈0 |
| XGBoost por defecto con protocolo fijo | 0,92562 | 0,77342 | 0,00425 | 0,15220 |
| Mejor prueba de Optuna sin restricción, índice 30 | 0,86577 | 0,76733 | 0,00448 | 0,09845 |
| **XGBoost seleccionado, índice 34** | **0,81135** | **0,76312** | **0,00324** | **0,04824** |

Se completaron **40 de 40 pruebas**, sin fallidas; **27** cumplieron el gap
medio estricto. La búsqueda completa duró **142,38 segundos**, sin incluir
referencias, diagnósticos ni el intento interrumpido. Los índices empiezan
en cero. El índice 34 no significa que solo se ejecutaran 34 pruebas.

La selección **no mejora el F1 frente al modelo por defecto**: pierde
aproximadamente 1,03 puntos en la escala de 0 a 100, al reducir el gap de
15,22 a 4,82 puntos. Cumple el requisito acordado para la media, no un límite
por fold ni una garantía de generalización. En el fold 4, por ejemplo, el
gap ronda 5,58 puntos. La cercanía de la media al umbral exige prudencia:
otra partición o población podría no cumplirlo. No se aumentó el presupuesto
ni se alteró el criterio después de ver los resultados.

La mejor prueba sin restricción tiene peor F1 de validación que la referencia
por defecto: esa referencia no pertenece necesariamente al espacio explorado.
40 pruebas no garantizan encontrar un óptimo ni superar toda configuración.

### Diagnósticos reunidos fuera de fold sobre train

Accuracy del seleccionado: **0,76299**, es decir, 76,30 % de anuncios OOF
correctamente clasificados. F1 macro OOF: **0,76315**, distinto de la media
por fold de 0,76312. Errores extremos: **102 de 10.088 anuncios**, 1,011 %.

| Categoría real de precio | Precisión | Exhaustividad (recall) | F1 | Anuncios reales |
|---|---:|---:|---:|---:|
| Económico | 0,88183 | 0,88235 | 0,88209 | 3.366 |
| Medio | 0,65643 | 0,66835 | 0,66234 | 3.359 |
| Premium | 0,75212 | 0,73803 | 0,74501 | 3.363 |

En cada fila, precisión se refiere a las predicciones con esa etiqueta;
recall se refiere a los anuncios realmente pertenecientes a esa categoría.
No representan resultados en producción ni sobre el test reservado.

Matriz de recuentos: **filas reales; columnas predichas**:

| Categoría real / predicha | Económico | Medio | Premium |
|---|---:|---:|---:|
| Económico | 2.970 | 343 | 53 |
| Medio | 349 | 2.245 | 765 |
| Premium | 49 | 832 | 2.482 |

Parámetros exactos del candidato: `xgboost_details.json`, clave
`selected_params`, junto con `fixed_params`. No redondear esos valores
para intentar reproducir los resultados.

Verificado: notebook completo, seis celdas de código ejecutadas sin error,
dos matrices gráficas y 31 tests correctos. Los tests verifican coherencia,
presupuesto, selección, mapeo, respeto del test reservado y resultados
agregados; no demuestran rendimiento con usuarios reales.

Se revisaron visualmente las dos matrices: títulos, filas reales, columnas
predichas, recuentos y unidades legibles, sin recortes visibles. Se contrastó
el CSV con las salidas propuestas de regresión logística y LightGBM: coinciden
las 18 columnas, configuración de CV, snapshot y firma del archivo de datos.
Es una comprobación de contrato/metadatos; no se reentrenaron esos modelos ni
se ejecutó aquí su lector de comparación, que aún no está integrado en la base.
