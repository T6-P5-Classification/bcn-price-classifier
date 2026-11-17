# Informe técnico — Barcelona Airbnb Price Classifier

**Versión para revisión del equipo (#38), 17/11/2026.** Síntesis del código y
los resultados integrados en `dev` (base `960e9c6`). El equipo ha seleccionado
XGBoost como champion. La evaluación final está preparada, pero todavía no
existe un resultado `champion_test.json` registrado en esta base. Este informe
no ejecuta test ni repite entrenamientos de otros responsables.

## 1. Problema y alcance

Clasificar un alojamiento de Barcelona en el segmento de su precio anunciado
por noche: Económico, Medio o Premium. El uso previsto es orientar al anfitrión
con características disponibles al publicar un anuncio. Premium significa
precio alto respecto al snapshot, no necesariamente lujo o calidad.

El sistema no estima un precio óptimo, ingresos, ocupación futura ni el
precio finalmente pagado. Las métricas de clasificación no cuantifican error
en euros ni demuestran una mejora económica para el anfitrión.

## 2. Fuente, preparación y trazabilidad

Fuente: [Inside Airbnb](https://insideairbnb.com/get-the-data/), Barcelona,
snapshot del **17/09/2026**, bajo
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Atribución: datos de Inside Airbnb; limpieza y transformaciones del proyecto.
La licencia de los datos no establece por sí sola una licencia del código.

El archivo principal es listings.csv.gz, conservado localmente en data/raw/;
no se incorpora a Git. El GeoJSON de barrios solo sirve para el mapa.
La limpieza y preparación se centralizan en
[src/cleaning.py](../src/cleaning.py) y
[src/preprocessing.py](../src/preprocessing.py).

El precio se convierte a valor numérico. Se descartan precios ausentes,
inválidos, no finitos o no positivos, porque no permiten asignar clase. Los
precios extremos positivos no se eliminan por defecto. Los nulos de las
entradas se conservan para imputarlos dentro de cada fold, salvo reglas
semánticas de limpieza documentadas. No se incorporan identificadores,
campos personales, textos libres ni derivados del precio a las entradas.

## 3. Target y split compartidos

load_train_test() aplica limpieza, feature engineering, selección de entradas,
split 80/20 y construcción de las tres clases, con random_state=42. Para el
snapshot de referencia hay **10.088 filas de train**, con 23 entradas antes
del encoding. El notebook del modelo descarta los objetos de test y no los
inspecciona.

Los umbrales definitivos son los cuantiles 1/3 y 2/3 de los precios de train,
sin redondearlos al generar etiquetas:

| Clase real según precio | Regla |
|---|---|
| Económico | price <= lower |
| Medio | lower < price <= upper |
| Premium | price > upper |

En la ejecución de referencia, lower=122.0 y upper=249.72333333333327.
No son constantes que deban sustituir el cálculo de cuantiles. El orden
común de presentación es Económico, Medio, Premium.

**Salvedad del protocolo:** la partición se estratifica con bins provisionales
calculados sobre los precios del dataset limpio completo. Los umbrales del
target final sí se calculan exclusivamente con train y permanecen fijos en
CV. El split aleatorio no separa anfitriones ni periodos temporales.

## 4. Las 23 entradas

MODEL_FEATURES es una lista cerrada. Una columna nueva del raw no entra en
el modelo por aparecer en el archivo. La simplificación de 71 a 23 entradas
busca usar información disponible para un alojamiento nuevo y reducir la
complejidad del formulario. La justificación y ablaciones están en
[las decisiones comunes](decisiones.md).

| Grupo | Entradas |
|---|---|
| Localización (4) | neighbourhood_cleansed, neighbourhood_group_cleansed, latitude, longitude |
| Alojamiento/estancia (6) | property_type, room_type, accommodates, bedrooms, beds, minimum_nights |
| Baños (3) | bathrooms_count, is_shared_bathroom, is_private_bathroom |
| Servicios (10) | has_air_conditioning, has_heating, has_hair_dryer, has_parking, has_self_checkin, has_dedicated_workspace, has_tv, has_pool, has_elevator, has_dishwasher |

El historial de reseñas, ocupación, disponibilidad y anfitrión queda fuera
de la lista final. El precio define la respuesta, pero nunca es una entrada.
Las coordenadas se mantienen; el protocolo aleatorio no prueba que esa señal
se transfiera a otras zonas o fechas.

## 5. Preprocesamiento y prevención de leakage

Cada estimador recibe un Pipeline completo con build_preprocessor():

- Numéricas por dtype: mediana y StandardScaler.
- Categóricas: moda y OneHotEncoder, min_frequency=20,
  handle_unknown=infrequent_if_exist y salida densa.

Las categorías nuevas van al grupo infrecuente si existe; si no, ese campo
se codifica como ceros. La dimensión tras encoding puede variar entre folds.
No se fija la mediana, escala o vocabulario con todo train antes de la CV.
El Pipeline se clona y ajusta dentro de cada partición de entrenamiento.

## 6. Validación y criterio de comparación

Se usa StratifiedKFold(n_splits=5, shuffle=True, random_state=42) sobre las
mismas filas de train. Optuna maximiza el **F1 macro medio de validación**;
no utiliza el test reservado. Los presupuestos registrados son 20 trials para regresión logística y 40
para Random Forest, XGBoost y LightGBM. No constituyen un presupuesto
idéntico entre modelos, aunque comparten datos, folds y métrica.

F1 macro calcula F1 por clase y promedia las tres con el mismo peso.
Precision responde qué proporción de las predicciones de una clase es
correcta; recall responde qué proporción de sus ejemplos reales se detecta.
F1 es la media armónica de precision y recall. Accuracy cuenta aciertos
sobre el total, sin equilibrar explícitamente las clases.

El control de sobreajuste es gap = media_F1_train - media_F1_validación,
con límite estricto **<0.05 absoluto** (cinco puntos porcentuales, no un 5%
relativo). Un gap pequeño también puede ocurrir con rendimiento bajo.

Para diagnósticos, cross_val_predict produce una predicción out-of-fold
(OOF) por fila de train usando el Pipeline completo y los mismos folds.
Accuracy, métricas por clase, matrices y errores extremos proceden de OOF.
El F1 agregado OOF puede diferir del F1 medio por fold; la comparación usa
este último. No se escoge un champion por el score agregado OOF.

## 7. Baseline verificada: Dummy y regresión logística

DummyClassifier usa most_frequent. LogisticRegression usa lbfgs,
regularización L2 por defecto, max_iter=5000 y semilla 42. Optuna explora
C log-uniforme en [0.001,100] y class_weight entre None y balanced, mediante
TPE con semilla 42. Mejor configuración: C=0.5971896122019431,
class_weight=None.

| Modelo | F1 train CV | F1 validación CV | Desviación CV | Gap | Accuracy train OOF | F1 train OOF |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0.166791 | 0.166791 | 0.000068 | 0.000000 | 0.333664 | 0.166791 |
| LogisticRegression | 0.739585 | 0.729123 | 0.006091 | 0.010462 | 0.729679 | 0.729183 |

La baseline supera Dummy bajo este protocolo. El gap de regresión logística
es 1.05 puntos porcentuales, inferior al límite de cinco. Esto no demuestra
que sea el mejor modelo del equipo ni una garantía de rendimiento futuro.

### Métricas de regresión logística por clase: train OOF

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Económico | 0.8545 | 0.8669 | 0.8606 | 3366 |
| Medio | 0.6132 | 0.6112 | 0.6122 | 3359 |
| Premium | 0.7188 | 0.7107 | 0.7147 | 3363 |

### Matriz de regresión logística: train OOF

Filas: clase real según precio/umbrales. Columnas: clase predicha por el
modelo, en el mismo orden. Incluye las 10.088 filas de train.

| Clase real / Clase predicha | Económico | Medio | Premium |
|---|---:|---:|---:|
| Económico | 2918 | 382 | 66 |
| Medio | 437 | 2053 | 869 |
| Premium | 60 | 913 | 2390 |

Medio presenta el menor F1. Los errores extremos se definen como los ejemplos
realmente económicos predichos premium, más los realmente premium predichos
económicos, divididos por el total: 126/10.088 = **1.25%**. El diagnóstico no
asigna un coste económico a esos errores ni modifica el entrenamiento.

## 8. Comparación de los cuatro modelos y champion

Las siguientes cifras proceden de los CSV agregados integrados de cada
responsable. Se han contrastado formato, snapshot, hash y configuración de CV;
no se presentan sus búsquedas completas como reejecutadas para este informe.

| Modelo | F1 train CV | F1 validación CV | Std CV | Gap (puntos) | Accuracy OOF | F1 OOF |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.811355 | 0.763119 | 0.003241 | 4.82 | 0.762986 | 0.763147 |
| LightGBM | 0.812453 | 0.762985 | 0.001777 | 4.95 | 0.762787 | 0.763016 |
| RandomForest | 0.805477 | 0.759354 | 0.005580 | 4.61 | 0.758822 | 0.759376 |
| LogisticRegression | 0.739585 | 0.729123 | 0.006091 | 1.05 | 0.729679 | 0.729183 |
| DummyClassifier | 0.166791 | 0.166791 | 0.000068 | 0.00 | 0.333664 | 0.166791 |

La regla registrada en #22 elige el mayor F1 macro medio de validación entre
modelos con gap < 0.05, excluyendo Dummy. El equipo confirmó **XGBoost** el
17/11/2026. XGBoost alcanza 0.763119 frente a 0.762985 de LightGBM: la
diferencia es aproximadamente **0.013 puntos porcentuales**, muy inferior a
la variación entre folds. Es un empate práctico, no evidencia de superioridad
estadística. Se mantiene la regla fijada, sin cambiar el criterio después
de observar resultados. La media de validación de XGBoost supera la de la
baseline logística en unos 3.40 puntos porcentuales.

### Configuraciones y presupuestos

| Modelo | Presupuesto | Configuración seleccionada resumida |
|---|---:|---|
| Regresión logística | 20 | C=0.5971896, class_weight=None, lbfgs, L2, max_iter=5000 |
| Random Forest | 40 | Prueba 8; configuración completa en rf_comparison.csv y rf_trials.csv |
| XGBoost | 40 | Prueba 34; 200 árboles, max_depth=6, learning_rate=0.1359633, regularización y muestreo registrados |
| LightGBM | 40 | 150 árboles, num_leaves=43, max_depth=6, learning_rate=0.0569220, regularización y muestreo registrados |

Los detalles completos permanecen en los artefactos de origen, evitando
duplicar aquí todos los parámetros. TPE y las semillas están documentados
en las guías de cada modelo. La regla de gap se aplica a la media, no exige
que cada fold individual quede por debajo del límite.

### Errores y diferencias de reproducción

El análisis integrado de #17 muestra que la confusión más frecuente en los
cuatro modelos es **real Premium, predicho Medio**. La clase Medio limita con
ambas clases y obtiene el F1 más bajo en la baseline, pero este patrón no
demuestra una causa. Los errores extremos de las métricas OOF originales son
126/10.088 en LogReg, 98/10.088 en RF, 102/10.088 en XGBoost y 98/10.088 en
LightGBM. Todos rondan el 1%, salvo LogReg que alcanza el 1.25%.

La reconstrucción local de RF en #17 difiere de su artefacto original por
siete aciertos (accuracy 0.759516 frente a 0.758822). El análisis conserva
ambos valores y la causa sin verificar. La tabla de selección de este informe
conserva las métricas originales de RF; no mezcla la reconstrucción con ellas.
Los otros tres modelos reproducen sus métricas OOF en ese análisis. Véase
[el análisis de errores](analisis_errores.md).

### Evaluación final preparada, todavía pendiente

`notebooks/08b_champion_test.ipynb` comprueba la selección, procedencia, esquema
y umbrales, ajusta el Pipeline elegido con todo train y prevé una sola
evaluación de test. El resultado se registra en `docs/results/champion_test.json`.
Ese archivo **no existe en la base revisada**. Por tanto, no se incluyen
accuracy, F1, informe por clase ni matriz finales de test, y no se afirma que
la evaluación ya haya ocurrido. No ejecutar este notebook como parte de la
revisión documental ni para preparar las diapositivas.

## 8.1. Aplicación y operación

La app básica integrada en `app/app.py` usa un formulario con las 23 entradas,
traducción de clases y un artefacto local con modelo y metadatos. El modelo
actual de la app es una **regresión logística provisional**, distinta de la
configuración champion seleccionada para la futura integración. El formulario
conserva las combinaciones observadas de tipo/modalidad y valida los campos
sin imputarlos silenciosamente. Las coordenadas proceden de las medianas por
barrio de train, no de una dirección personal.

La app muestra segmento, rango según umbrales de train y probabilidades
estimadas del modelo. No se ha demostrado calibración de esas probabilidades
ni mejora económica. La lógica de formulario se prueba con datos y modelos
sintéticos. Feedback persistente, métricas en tiempo real, reentrenamiento,
drift y despliegue siguen sus propias entregas. Este informe no acredita
MongoDB o una URL pública operativos.
## 9. Límites de la evidencia

- La lista de 23 variables también se seleccionó mediante ablaciones con
  CV sobre train, según decisiones.md. Además, la selección de
  hiperparámetros y los diagnósticos OOF usan los mismos folds. No es nested
  CV ni una evaluación final independiente: puede haber optimismo tanto por
  selección de variables como por ajuste de hiperparámetros.
- Versiones anteriores de la baseline ya mostraron resultados del mismo
  test reservado. Retirarlos no deshace esa exposición; el equipo debe
  considerarla antes de afirmar independencia de la evaluación final.
- El muestreo aleatorio no evalúa explícitamente nuevos anfitriones, barrios
  o snapshots. No se ha demostrado estabilidad temporal o ausencia de drift.
- Las clases se definen por precios anunciados del snapshot: no son clases
  universales de calidad ni niveles absolutos de precio para siempre.
- No se han acordado costes numéricos por clase ni umbral de aceptación del
  champion. No se inventan esos criterios a partir de los scores observados.
- La app básica está integrada con un modelo provisional. Este informe no
  acredita un despliegue público, integración del champion o persistencia de
  feedback; deben verificarse en sus entregas antes de presentarse como hechos.

## 10. Reproducción y procedencia

Para la preparación integrada en dev: seguir el [README](../README.md),
crear .venv con Python 3.12, instalar requirements.txt y descargar el snapshot
sin añadirlo a Git. Los notebooks de modelos usan directamente las funciones
compartidas y no necesitan ejecutar antes los notebooks de preparación.

La evidencia procede de los CSV `*_comparison.csv` y los JSON agregados de
[docs/results](results/), de las guías [LogReg](logreg.md),
[XGBoost](xgboost.md), [comparación](model_comparison.md) y
[decisiones](decisiones.md). Son resultados registrados de sus responsables.
Los notebooks `04` a `08` conservan las salidas de ejecución. No se vuelven a
entrenar todos los modelos para redactar este informe.

- Base documental comprobada: `960e9c6`.
- Snapshot: Barcelona 2026-09-17.
- SHA-256 del raw: `226f5e6af36cc82286765cda452cbe9abd6fbd70415aaf2b23a07805f6fbf23e`.
- CV común: cinco folds estratificados, shuffle=True, random_state=42.
- Procedencia y versiones por experimento: campos de cada CSV y JSON. No se
  supone que todas las ejecuciones compartieran exactamente el mismo entorno.

Para reproducir un candidato, seguir su guía y mantener el Pipeline dentro de
CV. Para inspeccionar la tabla, usar `08_comparison.ipynb`. No ejecutar `08b`
para una comprobación rutinaria, porque reserva la evaluación final. La app
provisional tiene sus instrucciones en el README. Los tests funcionales con
datos sintéticos no sustituyen una evaluación de calidad de un artefacto real.

## 11. Estado de entrega y próximos pasos

Este informe reúne el problema, datos, preparación, cuatro candidatos, regla
y decisión del champion, diagnóstico de errores y estado de la app. Su alcance
es documentar la evidencia disponible, sin bloquear la revisión del informe
por entregas todavía pendientes del resto del equipo.

| Elemento | Estado comprobado |
|---|---|
| Preparación y 23 entradas compartidas | Integrado |
| Cuatro candidatos y comparación CV/OOF | Resultados registrados e integrados |
| Champion | XGBoost, decisión registrada del equipo |
| Evaluación final de test | Notebook preparado, resultado no registrado |
| App básica | Integrada con regresión logística provisional |
| Integración del champion y feedback persistente | Pendiente de sus responsables |
| Despliegue público, drift y reentrenamiento | Sin evidencia operativa en esta base |
| Umbral mínimo de calidad (#32) | Valor de aceptación todavía no definido |

Al registrar la evaluación final, añadir sus métricas y conservar la advertencia
sobre exposición histórica de test. Al integrar champion, persistencia y
despliegue, actualizar este estado con evidencias. Las limitaciones no se
eliminan porque el modelo cumpla el gap. La revisión humana del equipo sigue
siendo necesaria antes de integrar el informe.
