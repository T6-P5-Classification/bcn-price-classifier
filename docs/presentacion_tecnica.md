# Presentación técnica: Barcelona Airbnb Price Classifier

Presentación en español de aproximadamente 20 minutos. Notas actualizadas con
la evidencia integrada en `dev`, base `fa1a9d4`, sin repetir la evaluación
del test reservado.

[Abrir Google Slides](https://docs.google.com/presentation/d/1zSvjH1LRa9Q_JK8cMZpOBaoV3WFl84wj65ChRRJQVkU/edit)

**Estado de esta revisión:** se revisa el guion versionado, no el contenido
actual de Google Slides. El enlace no concede acceso público. El equipo debe
contrastar y sincronizar las 16 diapositivas y sus notas antes de presentar;
no se han cambiado permisos ni escrito en Drive desde esta revisión.

Las referencias fijadas a `4b5f88d` conservan la procedencia histórica del
protocolo y los resultados. El estado operativo actualizado se contrasta con
las fuentes locales de las secciones 13 y 15, sobre la base `fa1a9d4`.

| Diapositiva | Tema | Tiempo orientativo |
|---|---|---|
| 1 | Airbnb Barcelona: Segmentos de precio | 0:45 |
| 2 | La pregunta del proyecto | 1:00 |
| 3 | Datos de Barcelona | 1:00 |
| 4 | Tres clases definidas con train | 1:30 |
| 5 | 23 entradas compartidas | 1:30 |
| 6 | Un Pipeline por fold | 1:15 |
| 7 | Validación sin consultar test | 1:30 |
| 8 | Cómo medimos el rendimiento | 1:30 |
| 9 | Cuatro modelos, una referencia | 1:00 |
| 10 | Resultados de validación | 1:30 |
| 11 | Champion: XGBoost | 1:30 |
| 12 | Errores del champion en train OOF | 1:30 |
| 13 | App con champion y feedback | 1:00 |
| 14 | Límites de la evidencia | 1:30 |
| 15 | Próximas entregas | 1:15 |
| 16 | Conclusiones y preguntas | 0:45 |

Los tiempos suman 20 minutos y requieren ensayo. Los candidatos se comparan
con CV/OOF de train. La diapositiva del champion distingue el resultado final
ya registrado en test: accuracy 0.764473 y F1 macro 0.765056. El historial de
exposición de test y las limitaciones se mantienen.

## 1. Airbnb Barcelona: Segmentos de precio

Proyecto 5 · Grupo 6
Beste, David, Esra y Miguel

### Notas para explicar

Presentamos bcn-price-classifier. El proyecto clasifica anuncios de Airbnb de Barcelona según su segmento de precio anunciado por noche. El equipo trabaja con una preparación compartida y compara cuatro familias de modelos. En esta charla explicaremos qué problema resolvemos, cómo evitamos usar información indebida, cómo interpretamos las métricas y por qué el equipo eligió XGBoost. Los resultados que mostraremos son de validación y predicciones fuera de fold sobre train. La evaluación final de XGBoost ya está registrada sobre 2.522 anuncios: accuracy 76,45% y F1 macro 0,765056. Distinguiremos ese resultado de la validación usada para elegir el modelo.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/decisiones.md)

## 2. La pregunta del proyecto

¿Qué segmento de precio corresponde al anuncio?
Económico, Medio o Premium
Características disponibles al publicar

### Notas para explicar

El usuario previsto es un anfitrión que introduce características de su alojamiento. El modelo devuelve un segmento de precio anunciado. Por ejemplo, puede situar un anuncio en Medio según sus características. Esto no calcula el precio óptimo ni garantiza reservas o ingresos. Premium significa que los precios de ese grupo superan el umbral del snapshot. No significa necesariamente mejor calidad o lujo. Conviene diferenciar la categoría real, obtenida del precio y los umbrales, de la categoría predicha a partir de las características. Esa distinción aparece después en las matrices de confusión.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/decisiones.md)

## 3. Datos de Barcelona

Inside Airbnb · 17 septiembre 2026
12.610 anuncios utilizables
Datos bajo licencia CC BY 4.0

### Notas para explicar

Usamos listings.csv.gz de Inside Airbnb. Después de la limpieza quedan 12.610 filas utilizables, repartidas en 10.088 para entrenamiento y 2.522 para el test reservado. Estas cifras describen este snapshot, no todas las viviendas de Barcelona. Los datos son observaciones de anuncios y no representan necesariamente transacciones reales. La licencia exige atribución. Conservamos el raw localmente, fuera de Git. Convertimos el precio a número y excluimos precios ausentes, inválidos, no finitos o no positivos. No eliminamos automáticamente todos los precios extremos positivos. La limpieza conserva el contrato del equipo.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/src/preprocessing.py)

## 4. Tres clases definidas con train

Económico: hasta 122 €
Medio: más de 122 € y hasta 249,7233… €
Premium: por encima de 249,7233… €

### Notas para explicar

Los límites son los cuantiles un tercio y dos tercios del precio de train. No son reglas universales de mercado ni valores fijados a mano. El código conserva toda su precisión: upper vale 249.72333333333327. Un precio igual al límite inferior es Económico y uno igual al superior es Medio. Aplicamos los mismos límites al test sin recalcularlos. El split es 80/20 con semilla 42. Hay una salvedad: la estratificación inicial usa bins provisionales calculados sobre todo el dataset limpio, aunque los umbrales finales sí usan train. También mantenemos esos umbrales fijos durante la CV. Explicar esta limitación forma parte de la transparencia del protocolo.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/src/preprocessing.py)

## 5. 23 entradas compartidas

Localización y características del alojamiento
Baños, estancia y servicios
El precio queda fuera de las entradas

### Notas para explicar

La primera selección era más amplia y el equipo la redujo a 23 variables antes del encoding. Son cuatro de localización, seis de alojamiento y estancia, tres de baños y diez de servicios. Podemos citar barrio, tipo de habitación, capacidad, dormitorios, camas y aire acondicionado. Se excluyen precio y sus derivados porque revelarían la respuesta. También se excluyen identificadores, datos personales e historial de reseñas o disponibilidad que un anuncio nuevo no tiene. La lista cerrada MODEL_FEATURES evita que una columna nueva entre por accidente. Las ablaciones de selección usaron CV sobre train, así que también pueden introducir optimismo en esa misma CV.

Fuente: [src/cleaning.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/src/cleaning.py)

## 6. Un Pipeline por fold

Imputación de valores ausentes
Escalado y codificación de categorías
Preprocesador y modelo se ajustan juntos

### Notas para explicar

El Pipeline reúne el preprocesamiento y el clasificador. Para variables numéricas usamos mediana y StandardScaler. Para categóricas usamos la moda y OneHotEncoder con min_frequency de 20 y manejo de categorías desconocidas. Todo se aprende dentro de cada fold de entrenamiento. Si calculáramos una mediana o el vocabulario usando también validación, estaríamos dando información de validación al entrenamiento. A eso lo llamamos leakage. Las 23 entradas pueden producir más columnas después del encoding y la dimensión puede variar entre folds. El estimador ve una representación numérica coherente con su propio entrenamiento.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/src/preprocessing.py)

## 7. Validación sin consultar test

Cinco folds estratificados · semilla 42
Optuna busca solo sobre train
Diagnósticos con predicciones OOF

### Notas para explicar

Dividimos train en cinco particiones, conservando aproximadamente las proporciones de clase. En cada ronda entrenamos con cuatro y validamos con la restante. Cada fila obtiene una predicción de un modelo que no la usó para ajustar ese fold: es una predicción out-of-fold, OOF. Optuna prueba hiperparámetros con esa CV. La regresión logística registra 20 trials y los otros tres modelos 40. No son presupuestos idénticos. Como seleccionamos parámetros y generamos OOF con los mismos folds, esto no es nested CV ni un test independiente. El test reservado corresponde solo a la evaluación final del champion.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/model_comparison.md)

## 8. Cómo medimos el rendimiento

Accuracy: proporción total de aciertos
Precision y recall: dos preguntas distintas
F1 macro: mismo peso para cada clase

### Notas para explicar

Accuracy divide los aciertos entre todos los ejemplos. Precision pregunta: de todos los anuncios que predije Premium, cuántos eran realmente Premium. Recall pregunta: de todos los anuncios realmente Premium, cuántos conseguí detectar. En la baseline logística, 2.390 de los 3.363 Premium reales se detectan como Premium: recall de 71,07%. F1 combina precision y recall mediante la media armónica. Macro calcula F1 por clase y promedia con el mismo peso para Económico, Medio y Premium. Elegimos por la media del F1 de validación de los folds. El F1 agregado de todas las predicciones OOF es otra cantidad, que puede diferir ligeramente.

Fuente: [docs/results/logreg_details.json](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/results/logreg_details.json)

## 9. Cuatro modelos, una referencia

Beste: regresión logística
Miguel: Random Forest
David: XGBoost
Esra: LightGBM

### Notas para explicar

DummyClassifier predice la clase más frecuente y ofrece una referencia sencilla. Nuestra regresión logística usa una relación lineal en las variables transformadas y regularización. Random Forest combina árboles mediante bagging. XGBoost y LightGBM construyen árboles mediante boosting, corrigiendo errores de forma secuencial. Cada responsable usa el split y el preprocesamiento comunes. La comparación se alimenta de todos los CSV de resultados compatibles en docs/results. El lector contrasta metadatos, añade procedencia y ordena por la métrica de validación. No hace falta que cada modelo modifique el código del lector.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/model_comparison.md)

## 10. Resultados de validación

Tabla de comparación CV y gap.

### Notas para explicar

Esta tabla usa la media de F1 macro de validación. XGBoost obtiene 0,763119, LightGBM 0,762985, Random Forest 0,759354 y regresión logística 0,729123. Dummy queda en 0,166791. Las desviaciones entre folds son aproximadamente 0,0032, 0,0018, 0,0056 y 0,0061 respectivamente. El gap se calcula como media F1 train menos media F1 validación. Lo expresamos en puntos porcentuales: XGBoost 4,82, LightGBM 4,95, RF 4,61 y LogReg 1,05. Todos cumplen la regla media inferior a cinco puntos, pero un gap bajo por sí solo no garantiza calidad. No confundir estos números con resultados finales de test.

Fuente: [docs/results/xgboost_comparison.csv](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/results/xgboost_comparison.csv)

## 11. Champion: XGBoost

Mayor F1 de validación con gap < 0,05
LightGBM queda en empate práctico
Test final: accuracy 76,45 % · F1 macro 0,7651

### Notas para explicar

El equipo confirmó XGBoost el 17 de noviembre siguiendo la regla registrada: elegir el mayor F1 macro medio de validación entre los candidatos con gap inferior a 0,05, excluyendo Dummy. La diferencia respecto a LightGBM es solo 0,000134 en escala F1, aproximadamente 0,013 puntos porcentuales. Es mucho menor que la variación entre folds, así que no demuestra superioridad estadística. Mantenemos la regla sin inventar un criterio después de ver los resultados. El resultado final ya está registrado en champion_test.json: sobre 2.522 anuncios, accuracy 0,764473 y F1 macro 0,765056, con 31 errores extremos (1,23%). El Pipeline se ajustó con todo train antes de esa evaluación. Solo leemos el registro existente, sin repetir la evaluación ni ajustar el modelo después. Se conserva la limitación de exposición histórica de test por la baseline. Fuente adicional: https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/results/champion_test.json

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/decisiones.md)

## 12. Errores del champion en train OOF

Matriz OOF de XGBoost: filas reales, columnas predichas.

### Notas para explicar

Las filas representan la categoría real según precio y umbrales. Las columnas representan la predicción. La diagonal contiene los aciertos. Por ejemplo, de los 3.363 Premium reales, 2.482 se predicen Premium, 832 Medio y 49 Económico. En los cuatro modelos la dirección de error más frecuente es real Premium y predicho Medio. En XGBoost los extremos son 53 Económicos predichos Premium más 49 Premium predichos Económicos: 102 de 10.088, aproximadamente 1,01%. Este diagnóstico no demuestra por qué ocurren los errores ni les asigna un coste en euros. RF presenta una reconstrucción local distinta de su métrica original por siete aciertos y el informe conserva esa salvedad.

Fuente: [docs/results/xgboost_details.json](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/results/xgboost_details.json)

## 13. App con champion y feedback

Formulario con las 23 entradas
Segmento y probabilidades estimadas
Champion XGBoost y feedback en Atlas

### Notas para explicar

La app Streamlit está integrada en dev y la demo pública está en https://bcn-price-classifier.onrender.com. El formulario pide características del alojamiento y deriva distrito y coordenadas de referencia por barrio para completar las 23 entradas del modelo. Muestra la categoría predicha, un rango según los umbrales y probabilidades estimadas, cuya calibración no se ha demostrado. El artefacto distribuido contiene solo el champion XGBoost, con comprobaciones de parámetros y procedencia: no hay experimento A/B en la demo. Con MONGODB_URI se guardan predicciones y respuestas en MongoDB; sin esa configuración se usan CSV locales. Un fallo de MongoDB no provoca un cambio silencioso a CSV. La demo usa Render y Atlas; la guía documenta la prueba ficticia y persistencia confirmadas anteriormente por el usuario, no una comprobación remota nueva de esta revisión. Abrir el formulario solo lee; pulsar Predecir o Enviar escribe registros. Para exponer, usar ejemplos ficticios y obtener autorización antes de guardar. Render Free puede tardar en despertar. Los tests sintéticos no sustituyen la prueba del servicio real.

Fuentes del estado actual: [app](../app/app.py), [almacenamiento](../src/feedback_store.py),
[despliegue y evidencias](render.md) y [manifiesto del modelo](../models/app_model.manifest.json).

## 14. Límites de la evidencia

Selección y OOF reutilizan los mismos folds
El test tuvo exposición histórica
Falta evaluar estabilidad en datos nuevos

### Notas para explicar

La lista de variables y los hiperparámetros se eligieron con CV sobre train. Reutilizar esos folds puede dar resultados optimistas. Además, versiones antiguas de la baseline mostraron resultados del mismo test reservado. Retirarlos no borra esa exposición. El informe lo declara para no presentar una independencia falsa. El split aleatorio tampoco separa anfitriones o fechas, por lo que no demuestra generalización temporal o a nuevos anfitriones. Los precios son anunciados y corresponden a un snapshot concreto. Un gap pequeño no sustituye esas comprobaciones. Las métricas actuales no prueban impacto económico ni que el modelo se mantenga estable en otro mercado.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/decisiones.md)

## 15. Próximas entregas

Recoger feedback y vigilar datos nuevos
Ensayar y sincronizar las diapositivas
Reentrenamiento automático y MLP pendientes

### Notas para explicar

La evaluación final está registrada y la app con champion único está desplegada. MongoDB, métricas de feedback, preparación de datos nuevos, controles de calidad y herramientas de drift ya están integrados. Las métricas del panel describen respuestas voluntarias con precio declarado, no a todos los anuncios ni precios verificados. El análisis de drift registrado compara junio con septiembre sobre train; no prueba vigilancia de datos futuros. El soporte A/B requiere otro artefacto con challenger y suficientes respuestas antes de comparar: no está activo en la demo actual. El reentrenamiento automático y el notebook MLP siguen vacíos, no operativos. La recogida de feedback no sustituye automáticamente el modelo. Los siguientes pasos son obtener evidencia de uso, revisar estabilidad con nuevos snapshots y ensayar la presentación, sin cambiar el modelo por resultados ya vistos de test. No se confunde una implementación con su ejecución remota verificada.

Fuentes del estado actual: [decisiones](decisiones.md), [métricas de feedback](feedback_metrics.md),
[drift registrado](results/drift_2026-06-24.json), [reentrenamiento](../src/retrain.py)
y [notebook MLP](../notebooks/09_model_mlp.ipynb).

## 16. Conclusiones y preguntas

Un protocolo compartido para comparar modelos
XGBoost seleccionado, con límites explícitos
Resultados y código reproducibles

### Notas para explicar

La aportación principal es una comparación trazable bajo un contrato común: mismas entradas, split y folds, preprocesamiento dentro del Pipeline y resultados claramente rotulados. Los cuatro modelos mejoran la referencia Dummy. XGBoost queda seleccionado por la regla registrada, con LightGBM prácticamente empatado. Conservamos las limitaciones de selección y exposición histórica de test. El repositorio contiene las guías y los artefactos agregados para revisar los resultados. Podemos cerrar preguntando qué evidencia de uso, estabilidad y mantenimiento necesita el equipo para operar la demo ya desplegada y abrir las preguntas de la audiencia.

## Verificación de la entrega

1. **Requisitos:** revisar la rama `docs/40-technical-presentation`, actualizada
   con `dev` en `fa1a9d4`. Disponer del repositorio, resultados agregados y,
   para pruebas locales, Python 3.12 con `requirements.txt`. No se necesitan
   credenciales de Atlas ni ejecutar notebooks de entrenamiento. Google Slides
   requiere acceso propio autorizado; pedirlo a la coordinadora si falta.
2. **Solo lectura local:** abrir desde README este guion. Comprobar que contiene
   las 16 secciones numeradas, notas y fuentes. Sumar los tiempos de la tabla:
   esperado 1.200 segundos (20 minutos orientativos, no duración ensayada).
3. **Solo lectura local:** contrastar las cifras de las secciones 10–12 con
   `docs/results/*_comparison.csv`, `xgboost_details.json` y
   `champion_test.json`. Distinguir CV, predicciones OOF de train y test final;
   las filas de las matrices son categorías reales y las columnas predichas.
   Revisar las secciones 13–15 junto con README y `docs/render.md`.
4. **Prueba local opcional:** desde la raíz, ejecutar
   `python -m pytest -q tests/test_deployment_artifact.py tests/test_docker.py`.
   Esperado: nueve tests correctos. Solo crea temporales/cachés locales; no
   escribe en Atlas, entrena ni vuelve a evaluar el test reservado.
5. **Solo lectura externa:** abrir Google Slides sin editar y comparar contenido,
   notas, tablas y fuentes con el guion actualizado. Esperado: misma evidencia
   y ningún estado antiguo de app provisional o despliegue pendiente. Esta
   comprobación no se ha ejecutado en la reparación local. Si hay diferencias,
   coordinar su actualización con la responsable y autorización específica
   antes de editar Drive; mientras tanto, no considerar el deck actualizado.
6. **Ensayo pendiente:** repartir intervenciones y cronometrar la exposición.
   Si excede el tiempo, simplificar tras revisión humana sin quitar las
   limitaciones de la evidencia. Ante cifras incoherentes o enlaces rotos,
   detener la aceptación y registrar el punto; no repetir entrenamientos,
   exponer secretos, enviar feedback remoto ni cambiar permisos para resolverlo.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/4b5f88d/docs/model_comparison.md)
