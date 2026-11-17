# Decisiones comunes de datos y modelado (#14)

Este documento describe el contrato compartido por los notebooks de modelos.
Las funciones de `src/cleaning.py` y `src/preprocessing.py` son la fuente de
verdad ejecutable: los notebooks las reutilizan, no duplican su lógica.

## Resumen: diez reglas para empezar un modelo

1. **Carga común:** usar `load_train_test()`; no repetir limpieza, variables
   derivadas, selección de columnas ni split en cada notebook de modelos.
2. **Target y entradas:** aprender los umbrales solo con train y reutilizarlos
   en test. No incluir `price` ni los campos excluidos en las features.
3. **Pipeline completo:** usar `build_preprocessor()` dentro del Pipeline del
   estimador; ajustar imputación, escalado y encoding en cada fold, no antes.
4. **CV común:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
   sobre las mismas filas de entrenamiento para todos los candidatos.
5. **Optuna sin test:** maximizar F1 macro medio de validación y registrar
   espacio de búsqueda, semillas, pruebas y tiempo; acordar el presupuesto.
6. **Métricas:** F1 macro principal, con `return_train_score=True`; añadir
   precision/recall/F1 por clase, accuracy, matriz de confusión y errores extremos.
7. **Sobreajuste y referencia:** registrar medias train/validación, dispersión
   de validación y gap menor que 0,05; comparar con Dummy. Un gap pequeño no
   basta para demostrar calidad.
8. **Test reservado:** elegir mediante validación, ajustar el candidato elegido
   con todo train y realizar la evaluación final en test, sin usarlo para ajustes.
9. **Reproducción:** registrar datos, código y dependencias; conservar Pipeline,
   esquema de entrada, etiquetas/mapeo y umbrales sin redondear con el modelo.
10. **Acuerdos verificables:** documentar decisiones, evidencias y limitaciones;
    distinguir resultados observados de hipótesis o propuestas pendientes.

El detalle y las salvedades de cada regla se desarrollan a continuación.

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
- `host_tenure_years` convive actualmente con `hosts_time_as_host_years` y `hosts_time_as_host_months` entre las features. Comparten información: la variable derivada combina años y meses (con el tratamiento de nulos definido en el código). Ninguna de estas variables entra en el modelo final (ver «Simplificación de variables»).
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

## Simplificación de variables: de 71 a 23

Tras la revisión del profesor («71 columnas son demasiadas») y el acuerdo del
equipo de orientar la app a **anuncios nuevos**, la selección se rehízo con un
criterio positivo: `MODEL_FEATURES` en `src/cleaning.py` es una **lista
cerrada de variables permitidas**. `select_model_features()` devuelve solo
esas columnas; una columna nueva del raw no entra en el modelo sin cambiar la
lista a propósito.

El análisis está en `notebooks/02_eda_classes.ipynb`, sección 6, y usa
**solo train**. Incluye una ablación reproducible: columnas exactas de cada
paso, resultados por fold y comparaciones laterales.

### Regla del producto (acordada por el equipo)

> Un anfitrión de Barcelona va a publicar un anuncio nuevo, describe su
> alojamiento y la app le dice si estará en el segmento Económico, Medio o
> Premium.

Por tanto, solo se usan datos que el anfitrión conoce **al publicar**. Lo que
aparece después (reseñas, ocupación, disponibilidad) no existe para un anuncio
nuevo. Esto acerca lo que ve el modelo al entrenar a lo que recibirá la app,
pero **no garantiza** el mismo rendimiento con usuarios reales: la población,
los datos ausentes y la forma de responder al formulario pueden cambiar.

### Ablación paso a paso (acumulativa)

F1 macro de validación, media de 5 folds estratificados (`random_state=42`),
solo train, `build_preprocessor()` y modelos por defecto
(`LogisticRegression(max_iter=3000)`, `HistGradientBoostingClassifier`).
Cada paso parte del conjunto del paso anterior.

| Paso | Columnas retiradas | Motivo | Nº | LogReg | Boosting |
|---|---|---|---|---|---|
| 0 | — | Todas las candidatas tras las exclusiones por reglas | 71 | 0,765 | 0,812 |
| 1 | Reseñas: `number_of_reviews*`, `review_scores_*`, `reviews_per_month`, `has_reviews`, `days_since_*_review`, `review_span_days` | **Producto:** un anuncio nuevo no tiene reseñas | 55 | 0,758 | 0,817 |
| 2 | Disponibilidad y ocupación: `availability_*`, `has_availability`, `estimated_occupancy_l365d` | **Producto:** aparecen después de publicar | 48 | 0,746 | 0,807 |
| 3 | Historial del anfitrión: `*listings_count*`, `hosts_time_as_*`, `host_tenure_years`, `host_has_profile_pic`, `host_identity_verified` | **Producto:** describen al anfitrión y su cartera, no el alojamiento | 36 | 0,740 | 0,796 |
| 4 | `beds_was_missing`, `minimum_nights_was_missing` | Describen huecos del dataset, no el alojamiento | 34 | 0,743 | 0,797 |
| 5 | Variantes de calendario de noches: `minimum_minimum_nights`, `maximum_minimum_nights`, `minimum_maximum_nights`, `maximum_maximum_nights`, `minimum_nights_avg_ntm`, `maximum_nights_avg_ntm` | **Simplificación con pérdida aceptada.** Algunas casi repiten `minimum_nights` en train (Spearman entre variables: `minimum_minimum_nights` 1,00; `minimum_nights_avg_ntm` 0,88; `maximum_minimum_nights` 0,82), pero en conjunto sí aportaban información | 28 | 0,730 | 0,783 |
| 6 | `host_is_superhost` | **Producto:** insignia basada en el historial (puntuación, estancias, cancelaciones) | 27 | 0,729 | 0,779 |
| 7 | `amenities_count` | Con las casillas `has_*` aporta poco; la app tendría que reproducir exactamente el recuento de Airbnb | 26 | 0,730 | 0,776 |
| 8 | `has_washer`, `has_clothes_dryer`, `maximum_nights` | **Pérdida aceptada** para simplificar el formulario (ver comparación 23 vs 26). `maximum_nights` vale en su mayoría valores por defecto (1125: 23,5 %; 365: 26 %; 999: 6 %) | **23** | **0,729** | **0,773** |

Lectura de la tabla:

- Los pasos 1–3 y 6 se deciden **por la regla del producto**, no por su
  aportación: el paso 1 no empeora (0,812 → 0,817), pero los pasos 2, 3 y 6
  cuestan unos 2 puntos en total con boosting. Es el precio de un modelo para
  anuncios nuevos.
- Los pasos 5, 7 y 8 son simplificaciones con **pérdidas pequeñas y
  aceptadas**. Que la correlación individual con la clase sea ~0 no demuestra
  ausencia de relación, especialmente para boosting, que puede usar efectos no
  lineales y combinaciones de variables.
- Son resultados **diagnósticos**: la selección se ha hecho mirando esta misma
  validación, por lo que son algo optimistas y no sustituyen la evaluación
  final en test.

### Comparaciones laterales (desde las 23 finales)

| Conjunto | Nº | LogReg | Boosting | Boosting por fold |
|---|---|---|---|---|
| 23 finales | 23 | 0,729 | 0,773 ± 0,006 | 0,770 · 0,769 · 0,784 · 0,769 · 0,775 |
| 23 + `has_washer`, `has_clothes_dryer`, `maximum_nights` | 26 | 0,730 | 0,776 ± 0,007 | 0,770 · 0,773 · 0,789 · 0,770 · 0,780 |
| 23 sin `latitude`/`longitude` | 21 | 0,728 | 0,761 ± 0,006 | 0,754 · 0,770 · 0,761 · 0,758 · 0,763 |

- **23 vs 26:** +0,3 puntos, menor que la variación entre folds. F1 por clase
  y errores extremos (Económico ↔ Premium, 0,95 % frente a 0,94 %)
  prácticamente iguales. Se mantienen las 23.
- **Coordenadas:** su correlación individual con la clase es ~0, pero quitarlas
  cuesta 1,2 puntos con boosting (efecto no lineal: zonas caras). Se mantienen.
- **Los 10 servicios `has_*`:** se compararon con una selección automática en
  train (1.872 servicios distintos → 1.060 normalizados → 81 presentes en el
  5–95 % de los anuncios → ordenados por correlación con la clase). Los 10
  elegidos en #9 dan un CV similar a los 10 primeros del ranking (0,775 frente
  a 0,778 en ese momento); con los 81 se ganaba +1,5 puntos.

⚠️ **Correlación no es causalidad.** Los servicios elegidos aparecen más en
los anuncios caros, pero no significa que *suban* el precio: muchos son
señales de otra cosa (el secador indica un piso totalmente equipado; el
ascensor, el tipo de edificio y la zona). Ayudan al modelo a **reconocer** el
segmento.

### Resultado

Con las 23 variables: F1 macro de validación 0,729 (LogReg) y 0,773
(boosting por defecto). Es el **F1 de validación con las 23 variables**, no
todavía una medida del rendimiento con usuarios reales. Con las 71 se obtenía
0,812, pero el formulario no podría rellenar 45 de esas columnas: imputándolas,
el boosting bajaba a 0,567 en la misma validación. El gap train-validación del
boosting sin ajustar (0,910 − 0,773 ≈ 14 puntos) supera el límite de 5: hay que
reducirlo con Optuna en #19–#21.

### Las 23 variables del modelo

| Grupo | Variables |
|---|---|
| 📍 Ubicación (4) | `neighbourhood_cleansed`, `neighbourhood_group_cleansed`, `latitude`, `longitude` |
| 🏠 Alojamiento (6) | `property_type`, `room_type`, `accommodates`, `bedrooms`, `beds`, `minimum_nights` |
| 🛁 Baños (3) | `bathrooms_count`, `is_shared_bathroom`, `is_private_bathroom` |
| ✨ Servicios (10) | `has_air_conditioning`, `has_heating`, `has_hair_dryer`, `has_parking`, `has_self_checkin`, `has_dedicated_workspace`, `has_tv`, `has_pool`, `has_elevator`, `has_dishwasher` |

13 de las 23 se crearon en feature engineering (#9) a partir de `amenities` y
`bathrooms_text`.

### Columnas excluidas por reglas (`MODEL_EXCLUDED_COLUMNS`)

Se mantiene como lista de privacidad y leakage; un test comprueba que ninguna
está en `MODEL_FEATURES`:

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

Antes de aceptar otro snapshot hay que revisar su esquema, privacidad y
posible información derivada del precio.

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
- El número de columnas codificadas puede variar entre folds. Las 94
  observadas al ajustar todo el train con las 23 variables no son una
  dimensión fija.

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

### Presupuesto acordado para Random Forest (#19)

Se fijan **40 pruebas de Optuna para Random Forest**, con las cinco particiones
comunes y solo entrenamiento: 200 ajustes de búsqueda. El espacio de búsqueda
es `n_estimators` 100–500 (paso 50), `max_depth` 3–20, `min_samples_split`
2–20, `min_samples_leaf` 1–10 y `max_features` 0,3–1,0. Parámetros fijos:
`random_state=42`, `n_jobs=1`, `bootstrap=True`, `verbose=0`. La semilla TPE
es 42; no hay poda, parada temprana ni ampliación automática. Referencias y
diagnósticos fuera de fold son ajustes adicionales, no nuevas pruebas de
Optuna.

Se maximiza F1 macro medio de validación y se selecciona la mejor prueba
completa que cumpla `gap < 0,05`. La mejor prueba sin restricción puede no
ser elegible. Si ninguna cumple, se informa que el requisito está pendiente,
sin evaluar test para corregirlo ni presentar otro candidato como válido.

Este acuerdo es específico de la entrega de Random Forest; no declara uniformes
los presupuestos anteriores del equipo ni autoriza repetir otras búsquedas.
La salida para la tabla común es un CSV independiente: esta issue no cambia
el notebook compartido de selección del champion ni utiliza test.

Resultado local comprobado de #19: 40 pruebas completas, 8 elegibles;
seleccionada la de índice 8. F1 macro medio de ajuste 0,80548 y validación
0,75935 (desviación poblacional 0,00558); gap medio 0,04612. Se cumple el
criterio para la **media**, no para cada fold ni para cualquier población.
Frente al Random Forest por defecto (validación 0,77789; gap 0,21126), se
acepta aproximadamente 0,0185 puntos menos de F1 para reducir el gap de unos
0,21 a menos de 0,05. Solo 8 de las 40 pruebas cumplen `gap < 0,05`: es un
buen ejemplo de por qué se usa el límite de gap. Accuracy OOF 0,75882 y tasa
extrema 0,00971; test sin evaluar. Parámetros, tiempos, versiones, métricas
por clase y límites quedan en el notebook y los resultados agregados.
### Presupuesto acordado para XGBoost (#20)

Se fijan **40 pruebas de Optuna para XGBoost**, con las cinco particiones
comunes y solo entrenamiento: 200 ajustes de búsqueda. El espacio de
búsqueda, parámetros fijos, semilla TPE 42, estados y tiempo total quedan
registrados en el notebook y los archivos agregados `docs/results/xgboost_*`.
No hay poda, parada temprana ni ampliación automática. Cualquier ampliación
debe acordarse antes de ejecutarla. Referencias y diagnósticos fuera de fold
son ajustes adicionales, no nuevas pruebas de Optuna.

Se maximiza F1 macro medio de validación y se selecciona la mejor prueba
completa que cumpla `gap < 0.05`. La mejor prueba sin restricción puede no
ser elegible. Si ninguna cumple, se informa que el requisito está pendiente,
sin evaluar test para corregirlo ni presentar otro candidato como válido.

Este acuerdo es específico de la entrega de XGBoost; no declara uniformes
los presupuestos anteriores del equipo ni autoriza repetir otras búsquedas.
El detalle reproducible y los límites están en [la guía de XGBoost](xgboost.md).
La salida para la tabla común es un CSV independiente: esta issue no cambia
el notebook compartido de selección del champion ni utiliza test.

Resultado local comprobado de #20: 40 pruebas completas, 27 elegibles;
seleccionada la de índice 34. F1 macro medio de ajuste 0,81135 y validación
0,76312 (desviación poblacional 0,00324); gap medio 0,04824. Se cumple el
criterio para la **media**, no para cada fold ni para cualquier población.
Frente a XGBoost por defecto (validación 0,77342; gap 0,15220), se acepta
aproximadamente un punto menos de F1 para reducir el gap. No se afirma una
mejora de puntuación. Accuracy OOF 0,76299 y tasa extrema 102/10.088; test
sin evaluar. Parámetros, tiempos, versiones, métricas por clase y límites
quedan en la guía específica y los resultados agregados.

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

## Contrato funcional de inferencia (#31)

`tests/test_model.py` prueba un Pipeline real con el preprocesador compartido,
23 entradas sintéticas y un estimador LogisticRegression de referencia.
Comprueba etiquetas, tamaño, probabilidades y su orden de clases, nulos,
categorías desconocidas, orden de columnas, columnas obligatorias y persistencia
joblib. La prueba de nulos confirma que inferencia no reaprende las medianas.
No cambia las funciones compartidas ni las decisiones de otros modelos.

Los tests sintéticos no certifican calidad predictiva ni sustituyen la prueba
del artefacto champion una vez integrado. Las seis pruebas comprueban este contrato con datos sintéticos. La suite
completa debe ejecutarse de nuevo al actualizar desde `dev`. Comandos y límites en
[docs/test_model.md](test_model.md).

## Modelo Random Forest (#19)

El notebook reutiliza el split y las 23 variables compartidas. Compara
Dummy most_frequent y Random Forest, usando siempre el Pipeline completo en
cinco folds estratificados (`shuffle=True`, `random_state=42`). Optuna
maximiza la media de F1 macro de validación con TPE y semilla 42, 40 trials
locales. Espacio: `n_estimators` 100–500 (paso 50), `max_depth` 3–20,
`min_samples_split` 2–20, `min_samples_leaf` 1–10 y `max_features` 0,3–1,0.
Parámetros fijos: `random_state=42`, `n_jobs=1`, `bootstrap=True`,
`verbose=0`. Se selecciona la mejor prueba completa con `gap < 0,05`; si
ninguna cumple, se informa que el requisito queda pendiente.

El Random Forest por defecto memoriza: F1 macro medio de ajuste 0,98915 y
validación 0,77789, con gap 0,21126. La mejor prueba sin restricción alcanza
0,77831 de validación, pero con gap 0,17641; memorizar no mejora la
validación de forma útil. El modelo elegido (índice 8) obtiene F1 macro medio
de ajuste 0,80548 y validación 0,75935 (desviación poblacional 0,00558),
con gap 0,04612, es decir, unos 0,0185 puntos menos de F1 que el modelo por
defecto a cambio de reducir el gap de 0,21 a menos de 0,05. Solo 8 de las
40 pruebas cumplen `gap < 0,05`.

## Baseline de regresión logística (#18)

El notebook reutiliza el split y las 23 variables compartidas. Compara Dummy
most_frequent y regresión logística, usando siempre el Pipeline completo en
cinco folds estratificados (`shuffle=True`, `random_state=42`). Optuna maximiza
la media de F1 macro de validación con TPE y semilla 42, 20 trials locales,
C log-uniforme entre 0.001 y 100 y class_weight None/balanced. Solver lbfgs,
L2 por defecto y max_iter=5000. El presupuesto común sigue pendiente.

El gap es media F1 macro train menos media F1 macro validación, con límite
estricto <0.05 absoluto. No se ajusta contra el test reservado para cumplirlo.
Tras elegir hiperparámetros, `cross_val_predict` produce las predicciones OOF
de cada modelo sobre train con los mismos folds y el Pipeline completo.
Accuracy, F1 macro, informes por clase, matrices y errores extremos usan solo
esas predicciones. El F1 agregado OOF no sustituye la media de F1 por fold
como criterio de selección. Las etiquetas se presentan en el orden común.
Accuracy OOF 0,75882, F1 macro OOF 0,75938 y tasa de errores extremos
0,00971.

El notebook no evalúa ni inspecciona los objetos de test: la evaluación del
test reservado corresponde solo al champion elegido en #22. No se serializa
un modelo ni se declara un champion. La tabla común contiene únicamente CV
y diagnósticos `train_oof`, con nombres de columnas `oof_*`, no `test_*`.
`rf_trials.csv` registra la búsqueda. No se guardan filas ni predicciones
individuales.

**Límites:** los hiperparámetros se seleccionan con los mismos folds: OOF no
es nested CV y sus diagnósticos pueden ser optimistas. El Random Forest por
defecto muestra un gap de 0,21; el límite de 0,05 es un requisito acordado,
no una garantía de buen servicio ni de ausencia universal de sobreajuste.
No se ha cambiado la configuración de búsqueda en respuesta a resultados de
test. La salida para la tabla común es un CSV independiente; esta issue no
cambia el notebook compartido de selección del champion ni utiliza test.

`tests/test_rf_model.py` ejecuta carga, CV, Optuna y diagnóstico con train
sintético y objetos de test que fallan si se inspeccionan. Comprueba que los
informes y las matrices cubren train y que no se exportan columnas de
resultados `test_*`. La ejecución completa verifica los artefactos reales.

## Importancia de variables (#16)

La importancia se calcula solo con train y validación cruzada, nunca con test.
Se usa permutación sobre cada fold de validación con el Pipeline completo y
F1 macro como métrica, agregando la media y desviación entre folds. El
resultado es una tabla por modelo con una fila por variable de
`MODEL_FEATURES`, más metadatos de snapshot, commit, hash y CV.

El método es model-agnostic y permite comparar importancias entre candidatos,
pero no establece causalidad: una variable puede ser sustituto de otra o
depender del preprocesamiento. La permutación es costosa y su estabilidad
depende de `n_repeats`; importancias negativas o cercanas a cero indican
variables con aportación nula o inestable en ese modelo.

La importancia del champion ya se ha generado para el XGBoost elegido en #22:
`docs/results/champion_feature_importance.csv` (con una línea inicial de
comentario que identifica el modelo). Se reutiliza el mismo protocolo y no se
incluyen filas del
dataset ni predicciones individuales; la tabla solo contiene métricas
agregadas y metadatos. `src/feature_importance.py` implementa el cálculo y
`tests/test_feature_importance.py` cubre su esquema y reproducibilidad.
`logreg_details.json` registra informes, matrices, folds, umbrales, versiones
y hashes; `logreg_trials.csv` registra la búsqueda. No se guardan filas ni
predicciones individuales.

**Límites:** los hiperparámetros se seleccionan con los mismos folds: OOF no
es nested CV y sus diagnósticos pueden ser optimistas. Versiones anteriores
de #18 ya evaluaron el mismo test; retirar sus resultados de la tabla no
deshace esa exposición histórica. Antes de presentar un resultado final
como independiente, el equipo debe considerar ese antecedente. No se ha
cambiado la configuración de búsqueda en respuesta a esos resultados.

`tests/test_logreg_notebook.py` ejecuta carga, CV, Optuna y diagnóstico con
train sintético y objetos de test que fallan si se inspeccionan. Comprueba
que los informes y las matrices cubren train y que no se exportan columnas
de resultados `test_*`. La ejecución completa verifica los artefactos reales.

### Carga automática de la tabla común (#22)

`08_comparison.ipynb` descubre todos los `docs/results/*_comparison.csv`,
comprueba su esquema y comparabilidad y los concatena sin reentrenar.
Añade `source_csv` para identificar el origen de cada fila y ordena por F1
macro medio de validación. Las referencias Dummy repetidas se conservan con
su procedencia; no se declara automáticamente un champion.

Se rechazan resultados de test, scores inválidos, gaps inconsistentes y
metadatos de snapshot/hash/CV diferentes. El contrato de columnas para los
productores está en [model_comparison.md](model_comparison.md). Las
instrucciones de ejecución de la baseline están en [logreg.md](logreg.md),
para mantener esta entrega independiente de las ediciones del README general.

## Tests de integridad de datos (#30): adaptación a las 23 variables

Los tests de `tests/test_data.py` emplean CSV comprimidos sintéticos y la
carga real `load_train_test()`, sin leer el snapshot ni evaluar su test
reservado. Se completa la tabla ficticia con barrio, distrito y tipo de
propiedad para cubrir todas las entradas actuales. Ambos conjuntos deben
coincidir exactamente con `MODEL_FEATURES`, incluido su orden, y conservar
los mismos tipos. Se mantienen comprobaciones independientes que prohíben
precio, identificadores, datos personales y campos derivados de ingresos:
usar la lista compartida no elimina esas protecciones.

Los nulos de `minimum_nights` deben conservarse en las filas correspondientes,
sin rellenado previo a la imputación del Pipeline. No se exige el indicador
`minimum_nights_was_missing` ni las antiguas entradas `amenities_count` o
`has_clothes_dryer`, que ya no pertenecen al modelo acordado. Se actualizan
expectativas de los tests, no las reglas de preparación ni el conjunto de
variables. La guía de ejecución está en el README y la verificación en
`docs/pruebas_manuales.md`, sección de la issue 30.

## LightGBM (#21)

El notebook `notebooks/07_model_lightgbm.ipynb` reutiliza el split y las 23
variables compartidas; el test se carga con la función común, pero no se usa
para evaluar ni seleccionar. Todas las validaciones
cruzadas usan el Pipeline completo, los mismos cinco folds y `n_jobs=1`.

- **Referencias:** Dummy `most_frequent` (F1 macro 0,167) y LightGBM por
  defecto (F1 validación ≈ 0,768, gap ≈ 0,11).
- **Optuna:** 40 pruebas secuenciales, TPE con semilla 42, solo train. Espacio:
  `n_estimators` 100–600, `learning_rate` 0,01–0,1 (log), `num_leaves` 8–48,
  `max_depth` 3–8, `min_child_samples` 20–150, `subsample` 0,6–1
  (`subsample_freq=1`), `colsample_bytree` 0,5–1, `reg_lambda` 0,001–30 (log).
  Maximiza la media de F1 macro de validación y registra el gap de cada prueba.
- **Selección:** la mejor prueba con gap < 0,05, la misma regla que XGBoost.
  Las pruebas con mejor validación (≈ 0,777) tienen gap de 0,14–0,19, así que
  se acepta ≈ 1,4 puntos menos de validación para cumplir el límite acordado.
- **Resultado (validación cruzada en train):** F1 train 0,812, validación
  0,763 ± 0,002, gap 0,049; errores extremos OOF 1,0 %. Por clase (OOF):
  Económico 0,88, Medio 0,66, Premium 0,75. Medio tiene el F1 más bajo; que
  limite con las otras dos clases es una posible explicación, no comprobada.
- **Tabla común:** `docs/results/lightgbm_comparison.csv`, con la
  configuración completa del estimador en `params`.
- **Registro completo:** `docs/results/lightgbm_trials.csv` (las 40 pruebas) y
  `docs/results/lightgbm_details.json` (versiones, regla de selección,
  resultados por fold y diagnósticos OOF).

**Límites:** el gap medio (0,0495) queda justo por debajo del límite y en 3 de
los 5 folds el gap individual lo supera ligeramente (máximo 0,052). La regla
común se aplica a la media, así que no se incumple, pero cumplirla no demuestra
que el modelo no sobreajuste; otra prueba con gap
0,043 y validación 0,761 es equivalente dentro del ruido entre folds, pero se
mantiene la regla común sin cambiarla tras ver los resultados. Seleccionar y
calcular OOF en los mismos folds no es validación anidada ni una evaluación
final.

## EDA geográfico y modalidad alquilada (#11)

El alcance confirmado utiliza solo entrenamiento de la carga común y las
categorías reales según precio/umbrales. Se realizan tres mapas de porcentaje
de categoría dentro de distrito, barras por distrito y `room_type`, y un
desglose distrito × modalidad con tamaños de muestra. No hay predicciones,
entrenamiento, cambios en las 23 variables ni consulta del test reservado.

El GeoJSON sirve solo para dibujar. Se comprueban nombres de barrio y
coherencia de distrito sin unir geometría a las entradas del modelo. Sus
polígonos de barrio se colorean con el porcentaje agregado del distrito,
con escala común 0–100 %, sin presentarlo como estimación por barrio. La
superficie del dibujo no mide número de anuncios. Las piezas repetidas de
un barrio se dibujan sin multiplicar registros.

Los porcentajes usan como denominador todos los anuncios del grupo indicado.
Se muestran recuentos y n; grupos observados con n < 30 se señalan como
pequeños mediante una convención descriptiva, no una prueba estadística.
Combinaciones sin observaciones conservan porcentajes ausentes y se pintan
grises, en lugar de confundirlas con 0 %.

Comprobado en el archivo local: 10.088 anuncios de train, 10 distritos,
67 barrios con correspondencia completa y cuatro modalidades. El GeoJSON
contiene 75 objetos y 73 nombres únicos, seis sin observaciones de train.
No se impone a entrenamiento la referencia previa de 68 barrios. Hay 40
combinaciones distrito × modalidad: siete vacías y 13 observadas con n < 30.
Son asociaciones descriptivas de este snapshot, no causas, rankings estables
de zonas pequeñas ni calidad predictiva demostrada. Reproducción, resultados
y límites: [EDA por distritos y room_type](eda_distritos_room_type.md).

## Tabla comparativa y champion (#22)

`notebooks/08_comparison.ipynb` reúne los `*_comparison.csv` de los cuatro
modelos (validación cruzada en train, mismos folds, snapshot y hash) y añade
una tabla legible para la presentación, con una sola fila Dummy.

**Regla de elección (acordada antes de ver los resultados):** mejor F1 macro
medio de validación entre los modelos con gap < 0,05, sin contar Dummy. El
notebook la aplica en código.

| Modelo | F1 val (media ± std) | Gap | Errores extremos (OOF) |
|---|---|---|---|
| **XGBoost** | **0,7631 ± 0,0032** | 0,048 | 1,0 % |
| LightGBM | 0,7630 ± 0,0018 | 0,0495 | 1,0 % |
| Random Forest | 0,7594 ± 0,0056 | 0,046 | 1,0 % |
| Regresión logística | 0,7291 ± 0,0061 | 0,010 | 1,2 % |
| Dummy | 0,1668 | 0,000 | 33,3 % |

**Champion: XGBoost**, confirmado por el equipo el 17/11/2026. LightGBM queda en empate práctico: la diferencia
(0,0001) es mucho menor que la desviación entre folds, y en los mismos cinco
folds cada uno gana en dos (uno igualado). Ambos serían defendibles; se
mantiene la regla fijada de antemano en lugar de elegir con un criterio
nuevo tras ver los resultados.

**Evaluación final:** `notebooks/08b_champion_test.ipynb` ajusta el Pipeline
de XGBoost (prueba 34 de Optuna, `src/xgboost_model.py`) con todo train y lo
evalúa **una sola vez** en test. Guarda `docs/results/champion_test.json` y se
detiene si ese archivo ya existe, tanto al principio como en la propia celda
de evaluación (antes de predecir); el archivo se crea en modo exclusivo y no
se sobrescribe. Antes de ajustar comprueba que los parámetros efectivos del
modelo son exactamente los de la fila ganadora del CSV de comparación. Después no se cambian modelo, variables ni
hiperparámetros. Se declara la exposición histórica del test por la baseline
de #18, que no guió esta elección. Antes de evaluar se comprobó, solo con
train, que el Pipeline reproduce la CV registrada (0,763119).
### Resultado de la evaluación única en test (#22)

Ejecutada una vez el 17/11/2026 con XGBoost ajustado en todo train
(`notebooks/08b_champion_test.ipynb`, registro en
`docs/results/champion_test.json`, código de `dev` en 960e9c6).

| | Validación cruzada (train) | Test |
|---|---|---|
| F1 macro | 0,7631 | 0,7651 |
| Accuracy | 0,7630 | 0,7645 |
| Errores extremos | 1,01 % | 1,23 % (31 de 2.522) |

Por clase en test: Económico F1 0,869, Medio 0,669, Premium 0,758. La
confusión más frecuente es Medio ↔ Premium (180 anuncios reales Medio
predichos Premium y 195 reales Premium predichos Medio).

En esta partición de test, el F1 macro observado (0,7651) es cercano al F1
macro medio de validación (0,7631). Es un resultado compatible con el
rendimiento estimado durante la selección, pero no demuestra ausencia de
sobreajuste ni rendimiento futuro. Se mantienen las limitaciones del reparto
aleatorio, el único snapshot y la exposición histórica del test. Tras ver el
resultado no se cambia el modelo, las variables ni los hiperparámetros.

## App Streamlit (#15)

La app (`app/app.py`) pide al anfitrión las 23 variables de `MODEL_FEATURES`
y muestra el segmento de precio anunciado estimado. La lógica del formulario
está en `app/app_logic.py` para poder probarla sin interfaz.

- **Nada se deduce si el usuario puede declararlo.** El tipo de alojamiento
  (`property_type`) y lo que se alquila (`room_type`) se preguntan por
  separado: en train el mismo tipo puede alquilarse entero o por
  habitaciones (p. ej. «Room in hotel»: 372 habitaciones privadas, 15 de
  hotel y 7 enteros). Las combinaciones que no aparecen en train se
  rechazan con un mensaje (p. ej. «Entire rental unit» como habitación
  compartida): el artefacto guarda `room_types_by_property` y los tipos
  ambiguos conservan todas sus modalidades observadas, sin imponer la más
  frecuente.
- **Baño:** se pregunta con tres opciones que reproducen los indicadores de
  `src/cleaning.py` sobre `bathrooms_text`: propio del alojamiento entero
  (0, 0), privado (0, 1) o compartido (1, 0). En train, todos los
  alojamientos enteros tienen (0, 0) y todas las habitaciones compartidas
  (1, 0); esas otras combinaciones se rechazan con un mensaje. En
  habitaciones privadas y de hotel también existe (0, 0) (216 y 6
  anuncios), pero solo significa que el texto original no decía «private»
  ni «shared», no que el baño fuera propio de un alojamiento entero. Por eso
  a una habitación se le pide aclarar si el baño es privado o compartido, en
  lugar de ofrecer una respuesta «sin especificar». Nada se corrige en
  silencio.
- **Rangos de precio:** se muestran con los umbrales de train redondeados
  al céntimo y la misma regla que el target: Económico ≤ límite inferior <
  Medio ≤ límite superior < Premium.
- **Sin imputación desde el formulario:** `build_row()` rechaza variables
  ausentes o vacías (comprobado después de convertir tipos, así detecta
  cualquier NaN), para que el Pipeline no las rellene sin avisar.
- **Barrio:** nombre oficial y distrito; latitud y longitud son la mediana
  de train del barrio (el formulario no pide la dirección exacta).
- **Etiquetas del modelo:** el artefacto guarda `class_labels`, que traduce
  las clases del modelo al nombre mostrado. Sirve tanto para modelos con
  etiquetas de texto (regresión logística) como numéricas 0/1/2 (XGBoost con
  `LABEL_TO_INT`), así que cambiar al champion no rompe la app.
- **Aviso en pantalla:** mientras el modelo era provisional (regresión
  logística) se indicaba así; desde #23 la app usa el champion (ver abajo). El resultado estima un segmento de precio anunciado; no
  recomienda un precio ni garantiza la clasificación. Se muestra la
  «probabilidad estimada por el modelo», no una «confianza».
- **Tests (`tests/test_app.py`):** modelo sintético, sin snapshot. Comprueban
  carga y predicción (AppTest), correspondencia formulario → variables,
  reglas de habitación y baño, rechazo de valores vacíos y lectura de
  etiquetas de texto y numéricas. No miden la calidad de las predicciones.

**Límites:** `models/app_model.joblib` se genera localmente y no se sube a
Git; la integración del champion y el despliegue van en #23 y #29/#33.
## Análisis de errores vecinos y extremos (#17): alcance y discrepancia documentada

El alcance confirmado analiza las cuatro configuraciones seleccionadas, sin
repetir Optuna, consultar test, cambiar variables ni elegir champion. Se añade
una sección separada en `08_comparison.ipynb`, conservando el lector existente.
Se mantienen las cinco particiones comunes y el Pipeline completo en cada
ajuste. Se distinguen las seis direcciones real/predicha, porcentajes sobre
todos los anuncios, sobre solo errores y dentro de cada categoría real.
No se atribuyen costes económicos ni se incluyen subgrupos o límites de precio.

La reconstrucción debe contrastarse con las métricas OOF de origen. La primera
ejecución se ha detenido con RF: accuracy local 0,7595162569389373 frente a
0,7588223632038065 guardada, siete aciertos de diferencia entre 10.088 anuncios.
Se verificaron datos, particiones, parámetros y coincidencia del código de RF
y preparación con su commit declarado; la causa aún no está verificada.
No se alteran el resultado de origen, el modelo o la tolerancia para forzar
coincidencias. Tras revisar procedencia y obtener la misma diferencia con las
funciones originales de RF, el usuario autoriza distinguir métricas de origen
y reconstrucción local. Solo RF tiene excepción explícita a la detención:
se conservan valores, diferencias y flags de coincidencia, sin presentar una
reproducción exacta. La causa sigue sin verificarse; no se atribuye al entorno
por observar una versión de Python distinta. El lector original conserva
sus métricas de selección. No se corrige otra entrega desde esta issue.
Evidencia, requisitos y límites: [análisis de errores](analisis_errores.md).

Diagnóstico local completado: 2.727 errores en regresión logística, 2.426 en
RF local, 2.391 en XGBoost y 2.393 en LightGBM, cada uno sobre los mismos
10.088 anuncios de train. De ellos son extremos 126, 98, 102 y 98,
respectivamente. Los demás son vecinos; no se invierten real y predicho ni
se mezclan porcentajes sobre todos los anuncios con porcentajes sobre solo
errores. La confusión más frecuente en los cuatro es real Premium/predicho
Medio. No se demuestra su causa, significación de diferencias pequeñas ni
se elige champion. No se guardan predicciones individuales ni se sustituye
la tabla de métricas original. RF sigue identificado como reconstrucción
local no idéntica al origen; los otros tres reproducen sus métricas OOF.

## Champion en la app (#23)

`src/train_app_model.py` entrena ahora el **champion XGBoost** de #22 en lugar
de la regresión logística provisional:

- Pipeline de `src/xgboost_model.py` con los hiperparámetros de la prueba
  seleccionada (`xgboost_details.json`). Antes de ajustar se comprueba que los
  parámetros efectivos coinciden con la fila ganadora de
  `xgboost_comparison.csv`; si no, se detiene.
- Ajuste con **todo train** y la codificación acordada (Económico=0, Medio=1,
  Premium=2); `class_labels` traduce las clases a sus nombres en la app. No se
  usa test para entrenar ni para elegir nada.
- **Mismos datos que en la selección:** antes de ajustar se comprueba que la
  firma del archivo de datos (`raw_sha256`), el tamaño de train y los umbrales
  coinciden con los registrados para el champion; si no, se detiene.
- El aviso bajo el resultado indica el modelo y, si existe
  `champion_test.json`, su resultado en la evaluación única de test
  (accuracy y F1 macro). Esas métricas solo se incorporan si el registro
  corresponde a este modelo: mismo champion, prueba de Optuna, parámetros,
  firma de datos, tamaño de train y umbrales. Si no coincide, el
  entrenamiento se detiene para no presentar resultados de otro modelo como
  si validaran el actual; si no existe, la app funciona sin mostrarlas. La
  procedencia se guarda en el artefacto (`provenance`). Se mantienen los límites: estima un segmento de
  precio anunciado, no recomienda un precio ni garantiza la clasificación.
- Tests: parámetros del champion, parada si difieren de la comparación,
  parada si cambian datos, tamaño de train o umbrales, rechazo de métricas de
  test de otro modelo, parámetros, datos o umbrales, lectura opcional del
  resultado de test, aviso de champion frente a
  provisional y AppTest con un modelo de etiquetas numéricas.

**Pendiente en #23:** el sistema de feedback. `models/app_model.joblib` sigue
generándose localmente y no se sube a Git; su uso en Docker/Render se decide
en #29/#33.

## Predicciones y feedback (#23)

La app guarda **cada predicción** y, opcionalmente, el **precio real por
noche** que indica el anfitrión. El contrato completo está en
`docs/feedback.md`; el acceso se hace solo con `src/feedback_store.py`.

- **Por qué cada predicción y no solo el feedback:** #36 (drift) y #25
  (métricas) necesitan todas las entradas y salidas, y #27 pide guardar
  «predicciones y feedback».
- **Por qué el precio y no un «¿acertó?»:** con el precio se obtiene la clase
  real con la misma regla y umbrales que el target, y #24 puede recalcular
  umbrales con datos nuevos.
- **Dos tablas** (`predictions`, `feedback`) unidas por `prediction_id`, una
  respuesta como máximo por predicción. Cada predicción guarda sus umbrales,
  `model_name`, `model_version` y `variant` (para #35).
- **Almacenamiento:** CSV en `data/feedback/` (fuera de Git), carpeta
  configurable con `FEEDBACK_DIR`. #27 sustituye solo `_append`/`_read` por
  MongoDB; las funciones públicas y los campos no cambian.
- **Una respuesta por predicción, también con envíos simultáneos:** la
  comprobación y la escritura se hacen bajo un bloqueo de archivo portable
  (macOS, Linux, Windows); un bloqueo abandonado por un proceso caído se
  ignora tras 60 s. Las probabilidades se validan (finitas, entre 0 y 1,
  suman 1) antes de escribir.
- **Errores de almacenamiento:** se convierten en `StorageError`, común a CSV
  y a la futura implementación MongoDB. La app lo captura: si falla guardar
  la predicción, la muestra igualmente sin formulario de respuesta; si falla
  guardar la respuesta, mantiene el resultado y muestra un aviso, sin
  confirmar el guardado. El resultado se conserva en pantalla al enviar la
  respuesta (`st.session_state`).
- **Aviso para el anfitrión:** bajo el resultado, una frase sencilla
  («Estimación orientativa… No es una recomendación de precio y puede no
  acertar»). Los detalles técnicos (modelo, resultado en test, F1 macro y
  límites) quedan en un desplegable «ℹ️ Sobre el modelo», para no mostrar
  jerga a quien usa la app sin ocultar la información.
- **Privacidad:** sin datos personales ni texto libre.
- **Tests:** `tests/test_feedback_store.py` (contrato de columnas, clase real
  en los límites, rechazo de precios y probabilidades no válidos, una
  respuesta por predicción también con cuatro envíos simultáneos, errores de
  escritura, bloqueo abandonado, unión) y AppTest del flujo predicción →
  respuesta, de los fallos de almacenamiento y del aviso sencillo.

**Límites:** respuestas voluntarias y no verificadas; en Render el disco no
es permanente, por lo que para conservar datos hace falta MongoDB.

## Dockerfile de la app (#29)

- **Base `python:3.12-slim`** (versión del proyecto) y `libgomp1`, que XGBoost
  necesita para OpenMP.
- **`requirements-app.txt`:** solo las dependencias de la app (pandas, numpy,
  scikit-learn, XGBoost, joblib, Streamlit) con las mismas versiones que
  `requirements.txt`. Se usa `xgboost-cpu`, la misma versión sin librerías de
  GPU: la imagen baja de 1,66 GB a 1,12 GB. Un test comprueba que las
  versiones coinciden con `requirements.txt`.
- **`.dockerignore` como lista de permitidos:** entran `app/`, `src/`,
  `.streamlit/`, el modelo y `requirements-app.txt`; quedan fuera datos
  (raw, processed, feedback), `.venv`, notebooks y docs.
  Después de las reglas de permitidos se excluyen expresamente secretos
  (`.env`, `.env.*`, `secrets.toml`, `*.pem`, `*.key`) aunque estén en
  carpetas permitidas; comprobado con archivos ficticios.
- **Modelo:** se copia `models/app_model.joblib`, que se genera en local con
  `python -m src.train_app_model` (no está en Git). Si falta, `docker build`
  falla al copiarlo.
- **Ejecución:** Streamlit se lanza con `exec`, así recibe la señal de
  parada y `docker stop` termina limpio (código 0, no 137). Usuario sin
  privilegios, puerto configurable con `PORT`
  (8501 por defecto; Render asigna el suyo), `HEALTHCHECK` en
  `/_stcore/health` y carpeta `data/` escribible para la app.
- **Comprobado en local:** la imagen se construye, el contenedor queda
  `healthy`, carga el champion, predice igual que en local (apartamento
  estándar → Medio, habitación con baño compartido → Económico, combinación
  contradictoria rechazada) y responde con `PORT=10000`.

**Pendiente para Render (#33):** Render construye la imagen desde GitHub,
donde no está el modelo. Hay que decidir en equipo cómo llega el modelo
(p. ej. versionar `models/app_model.joblib`, ≈ 1 MB, o descargarlo al
arrancar).

## Docker Compose: app + base de datos (#26)

`docker-compose.yml` arranca dos servicios:

- **`app`**: la imagen de #29. `FEEDBACK_DIR=/app/data/feedback` en el volumen
  `feedback-data`: las predicciones y respuestas (CSV de #23) se conservan al
  parar, recrear o reconstruir el contenedor. Arranca cuando MongoDB está sano
  (`depends_on: service_healthy`).
- **`mongo`**: `mongo:7.0.14` (versión exacta), datos en el volumen
  `mongo-data`, healthcheck con `mongosh` y **sin puerto publicado**: solo la
  app puede conectarse, por la red interna de Compose
  (`MONGODB_URI=mongodb://mongo:27017/bcn_price_classifier`).

**Alcance:** la app todavía no escribe en MongoDB; eso es #27, que solo tiene
que leer `MONGODB_URI` y sustituir el almacenamiento de
`src/feedback_store.py` (ver `docs/feedback.md`). Para uso local sin
autenticación; en un despliegue real (#28, MongoDB Atlas) la URI con
credenciales iría en un `.env` fuera de Git.

**Comprobado en local:** ambos servicios `healthy`, app con HTTP 200, la app
alcanza `mongo:27017`, MongoDB sin puertos publicados, y una predicción con
respuesta sigue guardada tras `docker compose down` y `up`. Tests estáticos
en `tests/test_docker_compose.py`.

## Tests de métricas mínimas (#32)

`src/model_quality.py` define los mínimos y `quality_problems()`, que devuelve
los que no se cumplen; `tests/test_metrics.py` los comprueba sobre los
resultados guardados (sin datos raw, sin reentrenar y sin volver a mirar el
test reservado).

| Métrica | Mínimo | Champion (CV / test) |
|---|---|---|
| F1 macro | ≥ 0,70 | 0,763 / 0,765 |
| F1 de cada clase | ≥ 0,60 | Medio 0,66 / 0,67 (la más baja) |
| Errores extremos (Económico ↔ Premium) | ≤ 3 % | 1,0 % / 1,2 % |
| Gap train − validación | < 0,05 | 0,048 / — |
| Margen sobre Dummy (F1 macro) | ≥ 0,30 | 0,60 / — |

Se comprueban el champion en validación cruzada y en la evaluación única de
test, que todos los modelos comparados superan el mínimo de F1, el gap y el
margen sobre Dummy, y, si existe, el artefacto de la app.

**Qué son estos mínimos:** una **alarma de regresión**, no un criterio de
selección (el champion se eligió en #22 con la regla acordada). Se fijaron
**después** de ver los resultados, algo por debajo de ellos, para detectar un
modelo claramente peor (p. ej. tras reentrenar en #34, que puede reutilizar
`quality_problems()` antes de sustituir el modelo) sin fallar por pequeñas
diferencias aleatorias. El gap mantiene el límite acordado de 0,05.

**Entradas inválidas:** `quality_problems()` no da por bueno un modelo con
métricas imposibles. Rechaza valores no numéricos, no finitos o booleanos
(`True` contaría como F1 = 1), F1 y tasas fuera de [0, 1] y un gap fuera de
[−1, 1]. Así, un error de cálculo no puede dejar pasar un modelo en #34.
