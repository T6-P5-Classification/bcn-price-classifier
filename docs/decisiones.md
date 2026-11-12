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
