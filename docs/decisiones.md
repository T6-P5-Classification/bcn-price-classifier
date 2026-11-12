# Decisiones comunes de datos y modelado (#14)

Este documento describe el contrato compartido por los notebooks de modelos.
Las funciones de `src/cleaning.py` y `src/preprocessing.py` son la fuente de
verdad ejecutable: los notebooks las reutilizan, no duplican su lógica.

## Objetivo y límites del producto

La aplicación ayuda al anfitrión a clasificar su alojamiento en un segmento
de **precio anunciado por noche**: `Económico`, `Medio` o `Premium`.
`Premium` significa precio alto respecto a los umbrales del snapshot, no
necesariamente lujo o calidad. No se predicen ingresos, ocupación futura,
precio finalmente pagado ni un precio óptimo.

Las clases se construyen por cuantiles y tienen tamaños aproximadamente
iguales; no se prioriza `Económico` por una supuesta mayor frecuencia.

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

- `amenities` se resume en `amenities_count` y en los indicadores `has_*` implementados en #9. No se generan indicadores específicos para `wifi`, `kitchen` ni `private_entrance`; siguen contando dentro de `amenities_count`. La justificación anterior citaba prevalencia y pruebas estadísticas sin una referencia reproducible en los notebooks versionados. Su escasa utilidad predictiva no se considera demostrada; cualquier revisión de esta selección debe contrastarse usando solo train.
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

### Umbrales y etiquetas reproducibles

Con el archivo local del snapshot indicado en el README y la configuración
por defecto, la comprobación da 10.088 filas de train y 2.522 de test.
Los umbrales calculados en train son `lower = 122.0` y
`upper = 249.72333333333327` euros por noche. Son resultados de referencia,
no constantes que deban copiarse al código. No redondear los umbrales para
construir etiquetas; el redondeo es solo de presentación.

El orden común de clases es `Económico`, `Medio`, `Premium`. Si un algoritmo
necesita enteros, usar el mapeo explícito `Económico: 0`, `Medio: 1`,
`Premium: 2` y conservar su inverso para interpretar predicciones.
No asumir que un codificador alfabético respeta este orden.

Los umbrales se fijan con el train completo antes de la validación cruzada;
los folds evalúan esa misma definición de clases, no aprenden umbrales
nuevos. Esto no es una evaluación temporal ni garantiza generalización a
otros barrios, anfitriones o snapshots.

## Inventario de columnas del modelo

`select_model_features()` elimina las columnas de `MODEL_EXCLUDED_COLUMNS`
cuando están presentes. La lista explícita vigente es:

```text
id, listing_url, scrape_id, name, picture_url,
host_id, host_url, host_profile_id, host_profile_url, host_name, host_about,
host_thumbnail_url, host_picture_url, host_neighbourhood, host_location,
license, neighbourhood, neighborhood_overview,
host_since, host_response_time, host_response_rate, host_acceptance_rate,
host_total_listings_count, host_verifications,
bathrooms, instant_bookable, source, description,
amenities, bathrooms_text, first_review, last_review,
price, price_quote_checkin_date, price_quote_checkout_date,
price_quote_total_price, price_quote_price_per_night, price_quote_raw,
estimated_revenue_l365d, last_scraped, calendar_updated, calendar_last_scraped
```

No hay un borrado dinámico de cualquier columna vacía ni una lista cerrada
de columnas permitidas: una columna nueva del raw podría entrar en `X`.
Antes de aceptar otro snapshot hay que revisar su esquema, privacidad y
posible información derivada del precio.

Para el snapshot actual se verifican **71 features antes del encoding**:

| Grupo | Columnas |
|---|---|
| Anfitrión | `hosts_time_as_user_years`, `hosts_time_as_user_months`, `hosts_time_as_host_years`, `hosts_time_as_host_months`, `host_is_superhost`, `host_listings_count`, `host_has_profile_pic`, `host_identity_verified`, `host_tenure_years` |
| Localización y alojamiento | `neighbourhood_cleansed`, `neighbourhood_group_cleansed`, `latitude`, `longitude`, `property_type`, `room_type`, `accommodates`, `bedrooms`, `beds` |
| Estancias | `minimum_nights`, `maximum_nights`, `minimum_minimum_nights`, `maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights`, `minimum_nights_avg_ntm`, `maximum_nights_avg_ntm` |
| Disponibilidad e historial de ocupación | `has_availability`, `availability_30`, `availability_60`, `availability_90`, `availability_365`, `availability_eoy`, `estimated_occupancy_l365d` |
| Reseñas | `number_of_reviews`, `number_of_reviews_ltm`, `number_of_reviews_l30d`, `number_of_reviews_ly`, `review_scores_rating`, `review_scores_accuracy`, `review_scores_cleanliness`, `review_scores_checkin`, `review_scores_communication`, `review_scores_location`, `review_scores_value`, `reviews_per_month`, `days_since_last_review`, `days_since_first_review`, `review_span_days` |
| Recuento de anuncios del anfitrión | `calculated_host_listings_count`, `calculated_host_listings_count_entire_homes`, `calculated_host_listings_count_private_rooms`, `calculated_host_listings_count_shared_rooms` |
| Indicadores de limpieza | `beds_was_missing`, `minimum_nights_was_missing`, `has_reviews` |
| Amenities y baños | `amenities_count`, `has_air_conditioning`, `has_heating`, `has_washer`, `has_hair_dryer`, `has_clothes_dryer`, `has_parking`, `has_self_checkin`, `has_dedicated_workspace`, `has_tv`, `has_pool`, `has_elevator`, `has_dishwasher`, `bathrooms_count`, `is_shared_bathroom`, `is_private_bathroom` |

Son entradas del clasificador actual, no una garantía de que el usuario
pueda rellenarlas todas para un alojamiento nuevo. La app debe acordar cómo
tratar campos históricos no disponibles, sin inventar valores ni asumir
que una evaluación con historial equivale a evaluar un anuncio nuevo.

## Primera celda común de los notebooks de modelado

Ejecutar desde `notebooks/`, con la raíz del proyecto disponible en el path:

```python
import sys
sys.path.append("..")
from src.preprocessing import load_train_test, build_preprocessor

X_train, X_test, y_train, y_test = load_train_test()
```

No repetir lectura, limpieza, selección de columnas o split en cada modelo.
Los notebooks de preparación anteriores (#6–#9) son demostraciones de sus
etapas, no plantillas alternativas de entrenamiento.

## Preprocesamiento dentro del Pipeline (#13)

`build_preprocessor()` devuelve un `ColumnTransformer` sin ajustar:

- Columnas numéricas por dtype: imputación con mediana y `StandardScaler`.
- Resto de columnas por dtype: imputación con moda y `OneHotEncoder`, con
  `min_frequency=20`, `handle_unknown="infrequent_if_exist"` y salida densa.
  Las categorías desconocidas van al grupo infrecuente si existe; si no,
  se codifican como ceros en ese campo.
- El número de columnas codificadas puede variar entre folds. Las 146
  observadas al ajustar todo el train no son una dimensión fija.

Cada modelo utiliza `Pipeline([("prep", build_preprocessor()),
("model", estimador)])`. Pasar el **Pipeline completo** a la validación y
a la búsqueda de hiperparámetros. Nunca ejecutar `fit_transform()` sobre
todo train antes de dividir los folds ni ajustar nada con test: mediana,
moda, escalado y categorías se aprenden dentro de cada fold de entrenamiento.

## Validación cruzada y selección de modelos

Usar la misma configuración y las mismas filas de train en todos los modelos:

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# pipeline contiene el preprocesador y el estimador aún sin ajustar.
scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
    return_train_score=True,
)
```

Aquí `scores["test_score"]` significa **validación del fold**, no el
conjunto reservado `X_test`. Cada fila pertenece una vez a la validación y
cuatro veces al entrenamiento de los cinco folds.

Optuna busca hiperparámetros (configuraciones del algoritmo) usando la media
de F1 macro de validación, con dirección de maximización. No utiliza test.
Registrar espacios de búsqueda, semilla cuando esté disponible, número de
pruebas y tiempo de búsqueda. El presupuesto común todavía debe acordarse
antes de ejecutar las comparaciones; no se fija un número arbitrario aquí.
Los resultados de la CV usada para optimizar pueden ser optimistas y no
equivalen a una evaluación final independiente.

Comparar candidatos por validación; considerar también estabilidad entre
folds, diferencia train/validación, coste de entrenamiento e inferencia.
Usar `DummyClassifier` como referencia con el mismo protocolo (#18).

### Métricas e interpretación

- **Principal: F1 macro**, media del F1 de las tres clases con el mismo peso.
  F1 combina precision y recall; no mide el error en euros ni distingue
  por sí solo la gravedad de confundir clases vecinas o extremas.
- **Por clase:** precision, recall, F1 y número de ejemplos. Para `Económico`,
  precision pregunta cuántos alojamientos predichos como económicos lo son;
  recall pregunta cuántos de los realmente económicos detectamos.
- **Complementarias:** accuracy y matriz de confusión. Cada matriz debe
  rotular filas como **clase real según precio y umbrales** y columnas como
  **clase predicha por el modelo**, con el orden común de las tres clases.
- Registrar la tasa de errores extremos: número de alojamientos realmente
  económicos predichos como premium, más los realmente premium predichos
  como económicos, dividido entre el total de predicciones evaluadas.
  Es un diagnóstico complementario, no sustituye la métrica principal.

Reportar media y desviación estándar del F1 macro de validación y la media
del F1 macro de entrenamiento. El criterio de control de sobreajuste es:

```text
gap = media_F1_macro_train - media_F1_macro_validación
gap < 0,05  (menos de cinco puntos porcentuales, no un 5 % relativo)
```

Una diferencia pequeña no demuestra buen rendimiento: ambos valores pueden
ser bajos. Una diferencia negativa debe informarse, no ocultarse; tampoco
prueba por sí sola ausencia de problemas. No ajustar contra test para lograr
este criterio ni presentarlo como garantía de calidad en producción.

## Uso del test reservado

El 20 % de test queda fuera del EDA que guía decisiones, selección de
features, búsqueda de hiperparámetros y elección del modelo. Elegir mediante
validación sobre train, ajustar el Pipeline elegido con todo train y
realizar una evaluación final planificada en test.

"Una vez" significa una evaluación final del modelo seleccionado con todas
las métricas acordadas; no limita a una sola métrica. Los análisis de errores
o importancia que vayan a guiar cambios deben realizarse con validación
(por ejemplo, predicciones fuera de fold), no consultando repetidamente test.
Si se cambia el modelo tras inspeccionar el resultado final, ese test ya
ha guiado decisiones: declarar la limitación y obtener una nueva evaluación
independiente antes de afirmar que la mejora está confirmada.

El split aleatorio actual no separa por anfitrión ni por tiempo. Sus métricas
describen este protocolo; no prueban rendimiento en anfitriones nuevos o
snapshots futuros. Los bins provisionales y la definición de target tienen
las salvedades documentadas en la sección de split.

## Comparación y reproducción de resultados

En la tabla común (#22), identificar algoritmo, configuración, versión de
datos/código, CV, F1 macro train y validación (con dispersión), gap, referencia
Dummy y tiempos. Separar resultados de validación de la evaluación final en
test; las métricas deben indicar siempre sobre qué conjunto se calcularon.

Para entregar el modelo a la app, conservar el Pipeline ajustado, orden de
columnas y tipos de entrada, etiquetas/mapeo, umbrales sin redondear, snapshot,
versiones de dependencias y configuración de entrenamiento. Esta sección fija
el contrato; guardar y cargar los artefactos corresponde a las issues de
modelos e integración, no añade esa implementación a #14.

## Costes de errores: propuesta pendiente de acuerdo

El uso previsto es orientar al anfitrión sobre el segmento de su alojamiento.
No se ha acordado favorecer una clase ni asignar costes numéricos. Mantener
F1 macro como criterio principal mientras el equipo decide esos pesos,
antes de ver qué candidato resulta favorecido.

La siguiente tabla describe posibles consecuencias, no pérdidas económicas
demostradas. **Filas: segmento real según el precio anunciado y los umbrales.
Columnas: segmento predicho por el modelo.**

| Segmento real / Segmento predicho | Económico | Medio | Premium |
|---|---|---|---|
| Económico | Clasificación correcta | El modelo asigna un segmento superior al real | El modelo asigna el extremo superior a un alojamiento del extremo inferior |
| Medio | El modelo asigna un segmento inferior al real | Clasificación correcta | El modelo asigna un segmento superior al real |
| Premium | El modelo asigna el extremo inferior a un alojamiento del extremo superior | El modelo asigna un segmento inferior al real | Clasificación correcta |

Sobreclasificar podría elevar expectativas e infraclasificar podría reducir
la percepción de valor, si el anfitrión usa la salida para orientar su precio.
Penalizar más los errores extremos es una propuesta, no una regla aprobada
ni una modificación del entrenamiento. No se asignan euros ficticios.

Como ampliación de producto, se propone mostrar mediana y rango de precios
anunciados de alojamientos comparables por barrio, tipo y segmento, indicando
fecha y tamaño de muestra. No es un segundo target ni una tasación. Queda
fuera del alcance de #14 y pendiente de acuerdo del equipo; tampoco se
incorpora un modelo de regresión de precios en esta entrega.
