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
- `property_type` se mantiene como variable categórica para el pipeline; no se agrupa en #9.
- La fecha de referencia para reseñas es configurable en `engineer_features(snapshot_date=...)`; por defecto usa `last_scraped` por fila para evitar fechas futuras.
- `bathrooms_text` se convierte en `bathrooms_count` (`Half-bath` = 0.5) y en `is_shared_bathroom` / `is_private_bathroom`.
- `host_tenure_years` se calcula a partir de `hosts_time_as_host_years/months`; en este snapshot `host_since` está vacío.
- `has_reviews` se conserva de la limpieza (#7) para distinguir anuncios sin historial.
- `days_since_last_review`, `days_since_first_review` y `review_span_days` usan por defecto `last_scraped` de cada fila; si se pasa `snapshot_date`, esa fecha se usa como referencia y las reseñas posteriores se convierten en `NaN`. Los nulos corresponden a anuncios sin reseñas.
- `clean_listings()` realiza solo limpieza; `engineer_features()` realiza feature engineering por separado.
