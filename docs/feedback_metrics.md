# Métricas del feedback en la app (#25)

En la app, abrir **Seguimiento del feedback** y activar **Mostrar métricas del
feedback**. El panel lee el almacenamiento compartido, se actualiza cada
10 segundos mientras la página está activa y permite **Actualizar ahora**.
No inicia tareas en segundo plano ni escribe predicciones o respuestas.
Desactivar la casilla deja de invocar el fragmento en los siguientes reruns.

## Fuente e integración

Se usan exclusivamente `load_labelled()` y `load_predictions()` de
[`src/feedback_store.py`](../src/feedback_store.py), según el
[contrato del equipo](feedback.md). No se leen CSV directamente ni se
modifica el backend de #27. La futura implementación MongoDB conserva esas
funciones públicas, columnas y `StorageError`.

La lectura obtiene primero las respuestas y después las predicciones: el
almacenamiento actual añade registros y no borra predicciones, por lo que
las respuestas leídas ya tienen una predicción en la segunda instantánea.
No se afirma que ambas lecturas sean una transacción atómica. Si el backend
cambia para permitir borrados, debe revisar esa garantía.

Cada grupo corresponde a `model_name`, `model_version`, `variant`,
`threshold_lower` y `threshold_upper`. La selección del panel muestra un solo
grupo: no mezcla artefactos, champion/challenger ni definiciones de clases.
Una versión registrada como «desconocida» conserva ese nombre; el panel no
puede distinguir artefactos que el productor haya guardado con la misma
versión desconocida y los mismos metadatos.

## Métricas y denominadores

- **Predicciones guardadas:** número de registros únicos del grupo.
- **Respuestas:** registros del grupo que tienen precio y clase real declarados.
- **Cobertura:** respuestas / predicciones guardadas del mismo grupo. No mide
  visitas o predicciones que no se pudieron guardar.
- **Distribución predicha:** cuenta las tres clases entre todas las predicciones
  guardadas, incluidas las que no tienen respuesta.
- **Accuracy, F1 macro y errores extremos:** usan únicamente las respuestas.
  Macro F1 da el mismo peso a las tres clases del contrato. Se usa
  `zero_division=0`, también cuando una clase tiene support cero.
- **Precision/recall/F1 y support por clase:** clase real obtenida del precio
  declarado con los umbrales de la predicción.
- **Matriz:** filas = clase real según precio declarado; columnas = clase
  predicha. Orden: Económico, Medio, Premium.
- **Error extremo:** real Económico/predicho Premium o real Premium/predicho
  Económico, dividido entre respuestas del grupo.

Sin predicciones se muestra un mensaje. Sin respuestas se muestran recuentos
y distribución, pero no accuracy ni F1. No se presenta cero como si hubiera
una evaluación real. Los fallos de lectura o registros inconsistentes
producen un aviso sin rutas, credenciales ni filas individuales.

## Controles del consumidor

El cálculo rechaza IDs duplicados, feedback sin predicción, clases desconocidas,
umbrales no finitos o desordenados y diferencias entre los metadatos/salida
de una respuesta y los de su predicción. Recalcula la clase del precio con
la función compartida y comprueba que coincida con `real_class`. No se basa
en un voto de acierto ni confía en la columna `correct` para calcular scores.
Solo devuelve agregados, sin IDs ni variables individuales del alojamiento.

## Pruebas

```bash
python -m pytest tests/test_feedback_metrics.py tests/test_feedback_panel.py -q
python -m pytest tests/test_app.py -q
python -m pytest -q
```

Los tests usan registros inventados y la API real del almacenamiento CSV en
carpetas temporales. Un ejemplo comprobado a mano tiene cinco predicciones,
cuatro respuestas, accuracy 0.5, macro F1 7/18 y un error extremo entre cuatro
respuestas. Se comprueban grupos, datos inválidos, falta de respuestas,
refresco tras guardar una respuesta y la conexión del panel en la app completa.
No se usan los datos reales de feedback, el test reservado ni un servidor
MongoDB para estas comprobaciones.

La suite del repositorio incluye además un test de procedencia del champion
que requiere el snapshot raw local. Mantenerlo ignorado por Git; no sustituirlo
por datos inventados ni eliminar el test para hacer pasar la suite.

## Límites

Las respuestas son voluntarias y el precio no se verifica. La muestra puede
ser pequeña o sesgada: el panel no certifica rendimiento futuro, impacto
económico, calibración ni sustituye la evaluación final del champion. No
vuelve a entrenar, ajusta hiperparámetros ni usa test. Los CSV locales requieren
un almacenamiento persistente para sobrevivir a redespliegues. El lector
actual carga todo el historial disponible; si crece mucho, el backend debe
acordar paginación o agregación sin cambiar el significado de los denominadores.
