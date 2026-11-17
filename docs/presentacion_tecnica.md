# Presentación técnica: Barcelona Airbnb Price Classifier

Presentación en español de aproximadamente 20 minutos. Las notas desarrollan
la explicación sin llenar las diapositivas. Base de resultados: `960e9c6`.

[Abrir Google Slides](https://docs.google.com/presentation/d/1zSvjH1LRa9Q_JK8cMZpOBaoV3WFl84wj65ChRRJQVkU/edit)

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
| 13 | La app básica ya está integrada | 1:00 |
| 14 | Límites de la evidencia | 1:30 |
| 15 | Próximas entregas | 1:15 |
| 16 | Conclusiones y preguntas | 0:45 |

Los tiempos son una guía de ensayo, no una duración medida. Ajustar con las
pausas, ejemplos y preguntas del equipo. Las métricas mostradas son CV/OOF de
train. La evaluación final de test sigue pendiente en la base documentada.

## 1. Airbnb Barcelona: Segmentos de precio

Proyecto 5 · Grupo 6
Beste, David, Esra y Miguel

### Notas para explicar

Presentamos bcn-price-classifier. El proyecto clasifica anuncios de Airbnb de Barcelona según su segmento de precio anunciado por noche. El equipo trabaja con una preparación compartida y compara cuatro familias de modelos. En esta charla explicaremos qué problema resolvemos, cómo evitamos usar información indebida, cómo interpretamos las métricas y por qué el equipo eligió XGBoost. Los resultados que mostraremos son de validación y predicciones fuera de fold sobre train. La evaluación final de test todavía está preparada, sin un resultado registrado en la base que hemos revisado.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/decisiones.md)

## 2. La pregunta del proyecto

¿Qué segmento de precio corresponde al anuncio?
Económico, Medio o Premium
Características disponibles al publicar

### Notas para explicar

El usuario previsto es un anfitrión que introduce características de su alojamiento. El modelo devuelve un segmento de precio anunciado. Por ejemplo, puede situar un anuncio en Medio según sus características. Esto no calcula el precio óptimo ni garantiza reservas o ingresos. Premium significa que los precios de ese grupo superan el umbral del snapshot. No significa necesariamente mejor calidad o lujo. Conviene diferenciar la categoría real, obtenida del precio y los umbrales, de la categoría predicha a partir de las características. Esa distinción aparece después en las matrices de confusión.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/decisiones.md)

## 3. Datos de Barcelona

Inside Airbnb · 17 septiembre 2026
12.610 anuncios utilizables
Datos bajo licencia CC BY 4.0

### Notas para explicar

Usamos listings.csv.gz de Inside Airbnb. Después de la limpieza quedan 12.610 filas utilizables, repartidas en 10.088 para entrenamiento y 2.522 para el test reservado. Estas cifras describen este snapshot, no todas las viviendas de Barcelona. Los datos son observaciones de anuncios y no representan necesariamente transacciones reales. La licencia exige atribución. Conservamos el raw localmente, fuera de Git. Convertimos el precio a número y excluimos precios ausentes, inválidos, no finitos o no positivos. No eliminamos automáticamente todos los precios extremos positivos. La limpieza conserva el contrato del equipo.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/src/preprocessing.py)

## 4. Tres clases definidas con train

Económico: hasta 122 €
Medio: más de 122 € y hasta 249,7233… €
Premium: por encima de 249,7233… €

### Notas para explicar

Los límites son los cuantiles un tercio y dos tercios del precio de train. No son reglas universales de mercado ni valores fijados a mano. El código conserva toda su precisión: upper vale 249.72333333333327. Un precio igual al límite inferior es Económico y uno igual al superior es Medio. Aplicamos los mismos límites al test sin recalcularlos. El split es 80/20 con semilla 42. Hay una salvedad: la estratificación inicial usa bins provisionales calculados sobre todo el dataset limpio, aunque los umbrales finales sí usan train. También mantenemos esos umbrales fijos durante la CV. Explicar esta limitación forma parte de la transparencia del protocolo.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/src/preprocessing.py)

## 5. 23 entradas compartidas

Localización y características del alojamiento
Baños, estancia y servicios
El precio queda fuera de las entradas

### Notas para explicar

La primera selección era más amplia y el equipo la redujo a 23 variables antes del encoding. Son cuatro de localización, seis de alojamiento y estancia, tres de baños y diez de servicios. Podemos citar barrio, tipo de habitación, capacidad, dormitorios, camas y aire acondicionado. Se excluyen precio y sus derivados porque revelarían la respuesta. También se excluyen identificadores, datos personales e historial de reseñas o disponibilidad que un anuncio nuevo no tiene. La lista cerrada MODEL_FEATURES evita que una columna nueva entre por accidente. Las ablaciones de selección usaron CV sobre train, así que también pueden introducir optimismo en esa misma CV.

Fuente: [src/cleaning.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/src/cleaning.py)

## 6. Un Pipeline por fold

Imputación de valores ausentes
Escalado y codificación de categorías
Preprocesador y modelo se ajustan juntos

### Notas para explicar

El Pipeline reúne el preprocesamiento y el clasificador. Para variables numéricas usamos mediana y StandardScaler. Para categóricas usamos la moda y OneHotEncoder con min_frequency de 20 y manejo de categorías desconocidas. Todo se aprende dentro de cada fold de entrenamiento. Si calculáramos una mediana o el vocabulario usando también validación, estaríamos dando información de validación al entrenamiento. A eso lo llamamos leakage. Las 23 entradas pueden producir más columnas después del encoding y la dimensión puede variar entre folds. El estimador ve una representación numérica coherente con su propio entrenamiento.

Fuente: [src/preprocessing.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/src/preprocessing.py)

## 7. Validación sin consultar test

Cinco folds estratificados · semilla 42
Optuna busca solo sobre train
Diagnósticos con predicciones OOF

### Notas para explicar

Dividimos train en cinco particiones, conservando aproximadamente las proporciones de clase. En cada ronda entrenamos con cuatro y validamos con la restante. Cada fila obtiene una predicción de un modelo que no la usó para ajustar ese fold: es una predicción out-of-fold, OOF. Optuna prueba hiperparámetros con esa CV. La regresión logística registra 20 trials y los otros tres modelos 40. No son presupuestos idénticos. Como seleccionamos parámetros y generamos OOF con los mismos folds, esto no es nested CV ni un test independiente. El test reservado corresponde solo a la evaluación final del champion.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/model_comparison.md)

## 8. Cómo medimos el rendimiento

Accuracy: proporción total de aciertos
Precision y recall: dos preguntas distintas
F1 macro: mismo peso para cada clase

### Notas para explicar

Accuracy divide los aciertos entre todos los ejemplos. Precision pregunta: de todos los anuncios que predije Premium, cuántos eran realmente Premium. Recall pregunta: de todos los anuncios realmente Premium, cuántos conseguí detectar. En la baseline logística, 2.390 de los 3.363 Premium reales se detectan como Premium: recall de 71,07%. F1 combina precision y recall mediante la media armónica. Macro calcula F1 por clase y promedia con el mismo peso para Económico, Medio y Premium. Elegimos por la media del F1 de validación de los folds. El F1 agregado de todas las predicciones OOF es otra cantidad, que puede diferir ligeramente.

Fuente: [docs/results/logreg_details.json](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/results/logreg_details.json)

## 9. Cuatro modelos, una referencia

Beste: regresión logística
Miguel: Random Forest
David: XGBoost · Esra: LightGBM

### Notas para explicar

DummyClassifier predice la clase más frecuente y ofrece una referencia sencilla. Nuestra regresión logística usa una relación lineal en las variables transformadas y regularización. Random Forest combina árboles mediante bagging. XGBoost y LightGBM construyen árboles mediante boosting, corrigiendo errores de forma secuencial. Cada responsable usa el split y el preprocesamiento comunes. La comparación se alimenta de todos los CSV de resultados compatibles en docs/results. El lector contrasta metadatos, añade procedencia y ordena por la métrica de validación. No hace falta que cada modelo modifique el código del lector.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/model_comparison.md)

## 10. Resultados de validación

Tabla de comparación CV y gap.

### Notas para explicar

Esta tabla usa la media de F1 macro de validación. XGBoost obtiene 0,763119, LightGBM 0,762985, Random Forest 0,759354 y regresión logística 0,729123. Dummy queda en 0,166791. Las desviaciones entre folds son aproximadamente 0,0032, 0,0018, 0,0056 y 0,0061 respectivamente. El gap se calcula como media F1 train menos media F1 validación. Lo expresamos en puntos porcentuales: XGBoost 4,82, LightGBM 4,95, RF 4,61 y LogReg 1,05. Todos cumplen la regla media inferior a cinco puntos, pero un gap bajo por sí solo no garantiza calidad. No confundir estos números con resultados finales de test.

Fuente: [docs/results/xgboost_comparison.csv](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/results/xgboost_comparison.csv)

## 11. Champion: XGBoost

Mayor F1 de validación con gap < 0,05
LightGBM queda en empate práctico
El test final todavía está pendiente

### Notas para explicar

El equipo confirmó XGBoost el 17 de noviembre siguiendo la regla registrada: elegir el mayor F1 macro medio de validación entre los candidatos con gap inferior a 0,05, excluyendo Dummy. La diferencia respecto a LightGBM es solo 0,000134 en escala F1, aproximadamente 0,013 puntos porcentuales. Es mucho menor que la variación entre folds, así que no demuestra superioridad estadística. Mantenemos la regla sin inventar un criterio después de ver los resultados. El notebook final está preparado para ajustar el champion con todo train y evaluar test una vez. En la base revisada no existe champion_test.json. Por eso no presentamos métricas finales de test.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/decisiones.md)

## 12. Errores del champion en train OOF

Matriz OOF de XGBoost con filas reales y columnas predichas.

### Notas para explicar

Las filas representan la categoría real según precio y umbrales. Las columnas representan la predicción. La diagonal contiene los aciertos. Por ejemplo, de los 3.363 Premium reales, 2.482 se predicen Premium, 832 Medio y 49 Económico. En los cuatro modelos la dirección de error más frecuente es real Premium y predicho Medio. En XGBoost los extremos son 53 Económicos predichos Premium más 49 Premium predichos Económicos: 102 de 10.088, aproximadamente 1,01%. Este diagnóstico no demuestra por qué ocurren los errores ni les asigna un coste en euros. RF presenta una reconstrucción local distinta de su métrica original por siete aciertos y el informe conserva esa salvedad.

Fuente: [docs/results/xgboost_details.json](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/results/xgboost_details.json)

## 13. La app básica ya está integrada

Formulario con las 23 entradas
Segmento y probabilidades estimadas
Regresión logística provisional

### Notas para explicar

La app Streamlit ya está en dev. El formulario recoge las características y muestra la categoría predicha, un rango de precio según los umbrales y probabilidades estimadas. El artefacto incluye el modelo y metadatos del formulario. El modelo de la app sigue siendo una regresión logística provisional, aunque XGBoost sea el champion elegido para la integración posterior. No debemos confundir selección del champion con despliegue efectivo. Los tests de la app usan un modelo sintético. No afirmamos que las probabilidades estén calibradas ni que exista una URL pública verificada. Podemos explicar o mostrar el formulario solo si el entorno local está preparado.

Fuente: [app/app.py](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/app/app.py)

## 14. Límites de la evidencia

Selección y OOF reutilizan los mismos folds
El test tuvo exposición histórica
Falta evaluar estabilidad en datos nuevos

### Notas para explicar

La lista de variables y los hiperparámetros se eligieron con CV sobre train. Reutilizar esos folds puede dar resultados optimistas. Además, versiones antiguas de la baseline mostraron resultados del mismo test reservado. Retirarlos no borra esa exposición. El informe lo declara para no presentar una independencia falsa. El split aleatorio tampoco separa anfitriones o fechas, por lo que no demuestra generalización temporal o a nuevos anfitriones. Los precios son anunciados y corresponden a un snapshot concreto. Un gap pequeño no sustituye esas comprobaciones. Las métricas actuales no prueban impacto económico ni que el modelo se mantenga estable en otro mercado.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/decisiones.md)

## 15. Próximas entregas

Evaluación final y artefacto del champion
Integración, feedback y control de calidad
Drift, reentrenamiento y despliegue

### Notas para explicar

El siguiente paso del protocolo es registrar la evaluación final autorizada sin reajustar el modelo después de verla. Después se integra el artefacto champion en la app y se desarrolla la persistencia del feedback. Las métricas de feedback necesitan etiquetas reales y una definición clara de qué anuncios se evalúan. También hay que acordar el mínimo de F1 de aceptación para el control de calidad. Drift, reentrenamiento y despliegue son entregas separadas y no se presentan como operativas sin evidencia. En documentación mantendremos el estado de cada componente actualizado, distinguiendo una implementación de sus pruebas y de su ejecución real.

Fuente: [docs/decisiones.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/decisiones.md)

## 16. Conclusiones y preguntas

Un protocolo compartido para comparar modelos
XGBoost seleccionado, con límites explícitos
Resultados y código reproducibles

### Notas para explicar

La aportación principal es una comparación trazable bajo un contrato común: mismas entradas, split y folds, preprocesamiento dentro del Pipeline y resultados claramente rotulados. Los cuatro modelos mejoran la referencia Dummy. XGBoost queda seleccionado por la regla registrada, con LightGBM prácticamente empatado. Conservamos las limitaciones de selección y exposición histórica de test. El repositorio contiene las guías y los artefactos agregados para revisar los resultados. Podemos cerrar preguntando qué comprobaciones necesita el equipo antes de usar el modelo en la app y abrir las preguntas de la audiencia.

Fuente: [docs/model_comparison.md](https://github.com/T6-P5-Classification/bcn-price-classifier/blob/960e9c6/docs/model_comparison.md)
