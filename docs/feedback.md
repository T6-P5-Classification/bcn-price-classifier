# Predicciones y feedback de la app (#23)

La app guarda **cada predicción** y, si el anfitrión lo indica, el **precio
real por noche** de su anuncio. Este documento es el contrato para las issues
que usan esos datos. **Usad siempre las funciones de `src/feedback_store.py`**,
no leáis ni escribáis los archivos directamente: así, cuando #27 pase el
almacenamiento a MongoDB, vuestro código no tendrá que cambiar.

## Qué se guarda

Dos tablas (hoy CSV en `data/feedback/`, fuera de Git; mañana colecciones de
MongoDB con los mismos campos):

### `predictions`: una fila por predicción mostrada

| Campo | Tipo | Significado |
|---|---|---|
| `prediction_id` | str | Identificador único (uuid4) |
| `created_at` | str | Fecha y hora UTC, ISO 8601 |
| `model_name` | str | Modelo usado, p. ej. `XGBoost` |
| `model_version` | str | Artefacto entrenado, p. ej. `xgboost-trial34-<commit>` |
| `variant` | str | `champion` (para #35 A/B: `challenger`) |
| 23 variables de `MODEL_FEATURES` | — | Entradas del modelo, mismos nombres y significado que en train |
| `predicted_class` | str | `Económico`, `Medio` o `Premium` |
| `prob_economico`, `prob_medio`, `prob_premium` | float | Probabilidades del modelo (suman 1) |
| `threshold_lower`, `threshold_upper` | float | Umbrales de train con los que se definieron las clases |

### `feedback`: como máximo una fila por predicción

| Campo | Tipo | Significado |
|---|---|---|
| `prediction_id` | str | Predicción a la que responde |
| `created_at` | str | Fecha y hora UTC, ISO 8601 |
| `real_price` | float | Precio real por noche (€) que escribe el usuario, > 0 |
| `real_class` | str | Clase de `real_price` con los umbrales de **esa** predicción y la misma regla que el target (`create_price_target`) |
| `correct` | bool | `real_class == predicted_class` |

No se guardan datos personales ni texto libre.

## Funciones

```python
from src.feedback_store import load_predictions, load_feedback, load_labelled

load_predictions()  # todas las predicciones
load_feedback()     # todas las respuestas
load_labelled()     # predicciones con respuesta: entradas + salida + precio y clase real
```

La carpeta por defecto es `data/feedback/`; se puede cambiar con la variable
de entorno `FEEDBACK_DIR` (tests, Docker) o con el argumento `directory=`.

## Garantías y errores

- **Una respuesta por predicción**, también con envíos simultáneos: la
  comprobación y la escritura se hacen bajo un bloqueo de archivo (válido en
  macOS, Linux y Windows). Un segundo envío da `ValueError`.
- **Validación antes de escribir:** las 23 variables, una clase conocida y
  probabilidades finitas entre 0 y 1 que suman 1; precio real > 0.
- **`StorageError`:** cualquier fallo de lectura o escritura (disco, permisos,
  archivo dañado) se convierte en este error común. La app lo captura: muestra
  igualmente la predicción y no confirma un guardado fallido. #27 debe lanzar
  el mismo `StorageError` desde MongoDB.

## Para cada issue

| Issue | Qué usar |
|---|---|
| **#24** Datos nuevos para reentrenar (`src/new_data.py`) | `load_labelled()`: las 23 variables y `real_price`. Se guarda el **precio**, no solo la clase, para poder recalcular umbrales con datos nuevos. |
| **#25** Métricas en tiempo real | `load_labelled()` para acierto (`correct`), F1 y matriz real/predicha; `load_predictions()` para número de predicciones y distribución de clases. |
| **#27** MongoDB (`src/database.py`) | Sustituir solo `_lock`, `_append` y `_read` de `src/feedback_store.py` por MongoDB (colecciones `predictions` y `feedback`, mismos campos). Mantener una respuesta por predicción (p. ej. índice único en `feedback.prediction_id`) y lanzar `StorageError` en los fallos. Las funciones públicas no cambian. |
| **#34** Reentrenamiento | Los datos de #24. |
| **#35** A/B testing | Implementado: la app guarda `variant` (`champion` o `challenger`) en cada predicción y `src/ab_test.py` compara el acierto por `variant` (`python -m src.ab_test`). |
| **#36** Data drift | `load_predictions()`: distribución de las 23 variables frente a train. |

## Límites

- Las respuestas son voluntarias y pocas al principio: las métricas de #25 no
  representan a todos los anuncios.
- El precio lo escribe el usuario y no se verifica.
- En Render (#33) el disco no es permanente: los CSV se pierden al
  redesplegar. Para conservarlos hace falta MongoDB (#27/#28).
