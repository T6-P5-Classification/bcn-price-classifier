## Selección de columnas y privacidad

Para el modelo se excluyen identificadores técnicos, URLs, nombres, textos
potencialmente personales, información del anfitrión que no describe el
inmueble y variables derivadas directamente de precio o ingresos.

`price` se conserva en el dataset limpio para crear el target, pero nunca se
usa como feature. `latitude` y `longitude` se conservan tanto para
visualizaciones como para el entrenamiento: pueden capturar diferencias de
precio a escala local. Su aportación se comprobará comparando modelos con y
sin coordenadas, para detectar posible sobreajuste geográfico.

Las vistas `model_df` y `map_df` se generan desde el mismo `cleaned_df`; no se
mantienen dos datasets físicos.

También se excluyen las columnas completamente vacías del snapshot, además de
`source` (75,75 % nulo y un único valor) y `description` (texto libre de alta
cardinalidad). `bathrooms` está vacía, pero `bathrooms_text` se conserva para
extraer variables en feature engineering.

## Feature engineering

- `amenities` se resume en `amenities_count` y en indicadores `has_*` solo para amenities con impacto práctico en el precio. Se excluyen `wifi` (prevalencia 97.9%, casi constante), `kitchen` (+0.4%, p=0.27) y `private_entrance` (-1.0%, p=0.87) por falta de impacto práctico.
- `has_dryer` se sustituye por `has_hair_dryer` y `has_clothes_dryer`, detectando cada amenity por separado para evitar confundir "Hair dryer" con una secadora de ropa.
- `property_type` se mantiene como variable categórica para el pipeline; no se agrupa en #9.
- La fecha de referencia para reseñas es configurable en `engineer_features(snapshot_date=...)`; por defecto usa `last_scraped` por fila para evitar fechas futuras.
- `bathrooms_text` se convierte en `bathrooms_count` (`Half-bath` = 0.5) y en `is_shared_bathroom` / `is_private_bathroom`.
- `host_tenure_years` se calcula a partir de `hosts_time_as_host_years/months`; en este snapshot `host_since` está vacío.
- `has_reviews` se conserva de la limpieza (#7) para distinguir anuncios sin historial.
- `days_since_last_review`, `days_since_first_review` y `review_span_days` usan por defecto `last_scraped` de cada fila; si se pasa `snapshot_date`, esa fecha se usa como referencia y las reseñas posteriores se convierten en `NaN`. Los nulos corresponden a anuncios sin reseñas.
- `clean_listings()` realiza solo limpieza; `engineer_features()` realiza feature engineering por separado.

## Split y target (#12)

- `load_train_test()` resuelve la ruta por defecto desde `src/preprocessing.py`,
  no desde el directorio de trabajo. Puede llamarse sin argumentos desde
  `notebooks/` o desde la raíz del repositorio.
- El split es 80/20, con `random_state=42`. Los bins provisionales de
  `pd.qcut(price, q=3)` usan los precios de todo el dataset limpio únicamente
  para estratificar la partición. Por tanto, test participa en la decisión
  de partición; estos bins no son features ni el target final del modelo.
- Los umbrales definitivos son los cuantiles 1/3 y 2/3 de `price_train`.
  Se aplican sin recalcular a test: `Económico` si `price <= lower`,
  `Medio` si `lower < price <= upper` y `Premium` si `price > upper`.
- El retorno habitual contiene cuatro objetos. `return_thresholds=True`
  permite recuperar los umbrales para guardarlos junto con el modelo.
  `price` nunca entra en las features; la imputación, el escalado y el
  encoding quedan fuera de #12.

## Baseline de regresión logística (#18)

`notebooks/04_model_logreg.ipynb` reutiliza el split y el Pipeline comunes.
Se compara `DummyClassifier(strategy="most_frequent")` con regresión logística
multiclase (`lbfgs`, L2 por defecto, `max_iter=5000`). Optuna maximiza F1 macro
medio con `StratifiedKFold(5, shuffle=True, random_state=42)`, pasando siempre
el Pipeline completo a cada fold. Se exploran `C` entre 0.001 y 100 en escala
logarítmica y `class_weight` entre `None` y `balanced`, con TPE y semilla 42.

Estas reglas de scoring y gap siguen la propuesta de decisiones todavía
pendiente de aprobación, no un acuerdo ya integrado en dev. El presupuesto
local de 20 trials tampoco representa un presupuesto común acordado.
El gap se calcula como F1 macro medio de train menos F1 macro medio de
validación; se exige `< 0.05` absoluto (cinco puntos porcentuales). La selección
maximiza validación; el gap se comprueba después, sin retocar contra test.

El test se consulta tras congelar la selección, para ambos modelos sobre las
mismas filas. Se guardan accuracy, F1 macro, informes por clase y matrices
rotuladas, junto a dispersión de CV, tiempos, parámetros y procedencia.
`docs/results/logreg_comparison.csv` alimenta `08_comparison.ipynb`;
`logreg_details.json` conserva informes, matrices, folds, umbrales completos,
versiones y hashes; `logreg_trials.csv` registra la búsqueda. No contienen
filas individuales del dataset. No se publica un modelo serializado.

La CV optimizada no es una estimación independiente. El split aleatorio no
separa anfitriones ni fechas; las métricas describen el snapshot y protocolo
actuales. Los bins provisionales globales y los umbrales fijos de train son
las salvedades documentadas en la sección de split.
