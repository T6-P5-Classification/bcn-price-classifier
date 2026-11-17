# Pruebas de funcionamiento del modelo (#31)

Desde la raíz, con Python 3.12 y las dependencias de `requirements.txt`:

```bash
python -m pytest tests/test_model.py -q
python -m pytest -q
```

Los seis tests usan datos sintéticos con las 23 columnas de `MODEL_FEATURES`
y un Pipeline real formado por `build_preprocessor()` y LogisticRegression.
No descargan datos, no consultan test reservado y no seleccionan un champion.

Se comprueban:

- Número de predicciones, tres etiquetas válidas y esquema sin precio.
- Probabilidades finitas, entre cero y uno, normalizadas por fila y coherentes
  con la clase predicha usando `model.classes_` como orden.
- Nulos de entrada sin modificar el DataFrame ni reaprender las medianas.
- Categoría no vista y columnas de entrada en distinto orden.
- Rechazo de una columna obligatoria ausente.
- Persistencia joblib en un directorio temporal: mismas clases, predicciones
  y probabilidades al recargar. Solo se carga el archivo creado por el test.

Estos tests verifican el contrato del Pipeline con un estimador de referencia;
no certifican la calidad predictiva ni el artefacto de producción de un
champion todavía no integrado. Al integrar ese artefacto debe probarse también
su carga e inferencia real. No cargar pickles/joblib de origen desconocido.

Validación de esta entrega: seis tests nuevos y 23 en la suite completa,
todos correctos. Se comprobó que el test de categoría desconocida falla al
sustituir temporalmente `handle_unknown` por `error`, y vuelve a pasar al
restaurar el código original. No se modifica código de producción.
