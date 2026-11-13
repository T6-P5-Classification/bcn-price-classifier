# Contrato de la tabla común (#22)

Desde la raíz, ejecutar `notebooks/08_comparison.ipynb` con su kernel del
proyecto (el notebook admite raíz o `notebooks/` como directorio de trabajo).
No entrena modelos ni evalúa test: lee automáticamente
`docs/results/*_comparison.csv` y ordena por F1 macro medio de validación.
Archivos como `*_trials.csv` no se incluyen.

Cada modelo puede aportar su propio archivo, por ejemplo
`lightgbm_comparison.csv`, sin modificar el notebook. Los nombres de columnas
deben coincidir entre los CSV; se admite distinto orden de columnas. El
formato común actual es el de `logreg_comparison.csv`:

| Columna | Significado |
|---|---|
| model | Nombre del algoritmo/configuración o de la referencia Dummy |
| cv_train_f1_macro | Media de F1 macro de entrenamiento de los cinco folds |
| cv_validation_f1_macro | Media de F1 macro de validación; criterio de ordenación |
| cv_validation_std | Desviación estándar poblacional del F1 macro de validación |
| gap | Media F1 train menos media F1 validación; puede ser negativo |
| gap_under_005 | Booleano consistente con gap < 0.05 |
| cv_fit_seconds | Suma de tiempos fit de los cinco folds de la configuración elegida |
| evaluation_set | Literal train_oof para los diagnósticos agregados |
| oof_accuracy | Accuracy agregada de las predicciones OOF sobre train |
| oof_f1_macro | F1 macro agregado OOF; no sustituye la media por fold |
| oof_extreme_error_rate | Errores reales Económico/predichos Premium y viceversa, divididos por n_train |
| oof_seconds | Tiempo completo de cross_val_predict, incluidos los fits y predicciones |
| search_seconds | Tiempo total de búsqueda; cero para Dummy sin búsqueda |
| params | Configuración, por ejemplo una cadena JSON |
| cv | Configuración común; StratifiedKFold(5, shuffle=True, random_state=42) |
| snapshot | Fecha del snapshot, en formato YYYY-MM-DD |
| code_commit | Commit base registrado por la ejecución; puede variar entre modelos |
| raw_sha256 | SHA-256 del listings.csv.gz utilizado; debe coincidir entre modelos |

Los CSV de comparación contienen métricas, configuración y procedencia, nunca
filas del dataset ni predicciones individuales. `source_csv` se añade solo al
mostrar la tabla: identifica el archivo de origen y no debe guardarse en los
CSV de cada modelo. Si varios modelos incluyen una referencia Dummy, se
conservan sus filas con ese origen visible; no se eliminan configuraciones
por compartir el mismo nombre.

## Validación al cargar

Se requieren archivos no vacíos, nombres de modelo y todas las columnas
comunes. Se rechazan columnas `test_*` y diagnósticos que no sean train_oof.
Los scores deben ser finitos y estar entre 0 y 1; el gap y su indicador deben
ser coherentes. Todos los resultados deben usar el mismo snapshot, hash de
datos y configuración de CV. Un archivo incompatible detiene la comparación
con un error: no se omite silenciosamente.

Esta validación comprueba metadatos declarados; no demuestra por sí sola que
cada productor ejecutó el protocolo correctamente. Hay que revisar también
sus notebooks, variables compartidas, presupuestos y evidencias de ejecución.
No se atribuyen resultados simulados de los tests a modelos reales.

## Champion y test reservado

La tabla facilita la comparación, pero no declara un champion automáticamente.
La selección y su justificación requieren los resultados comparables de los
cuatro modelos y revisión del equipo: media/dispersión de validación, gap,
referencia Dummy y costes. Solo el champion elegido se evalúa en test en #22.
OOF tras selección en los mismos folds no equivale a nested CV o a un test
independiente. Debe conservarse la advertencia sobre la exposición histórica
del test de la baseline.

## Comprobaciones

```bash
python -m pytest tests/test_comparison_notebook.py -q
python -m pytest -q
```

Los tests usan CSV sintéticos temporales para comprobar descubrimiento de
múltiples archivos, orden por validación frente a OOF, cambios de orden de
columnas y rechazo de formatos, scores, gaps o protocolos incompatibles.
No generan métricas de test reservado ni entrenan modelos del equipo.
