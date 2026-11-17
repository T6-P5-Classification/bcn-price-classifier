# EDA por clase: distritos y modalidad alquilada (#11)

## Entrega y alcance

- Rama: `feat/11-district-room-type-eda`, desde `dev` en `75f3491`.
- Notebook: `notebooks/03_eda_map.ipynb`, preparado para raíz o `notebooks/`.
- Funciones de agregación y dibujo: `src/eda_map.py`.
- Se utilizan únicamente las filas de entrenamiento de `load_train_test()`.
  Sus dos objetos reservados se descartan sin inspeccionar. La función común
  sí los carga; no se afirma que el test nunca se leyó en memoria.
- No se entrena un modelo, no se calculan predicciones y no se modifican las
  23 entradas. Las categorías **reales** proceden del precio y sus umbrales,
  no de un clasificador. Premium no significa lujo ni calidad.

La entrega incluye tres mapas en una figura, barras por distrito, barras por
modalidad y un desglose distrito × modalidad en tres paneles. Cada gráfico
explica población, unidades, denominador y límites de interpretación.

## Contrato del mapa

`neighbourhoods.geojson` se usa solo para dibujar. Se verifica por nombre
`neighbourhood_cleansed` frente a `properties.neighbourhood` y se comprueba
que coincida el distrito de `properties.neighbourhood_group`. No se realiza
una unión espacial, no se añaden columnas geométricas al modelo, no se
publican puntos de anuncios y no se genera un segundo dataset de registros.

Cada panel corresponde a Económico, Medio o Premium y muestra:

> Anuncios de esa categoría REAL en el distrito / todos los anuncios de
> entrenamiento de ese distrito × 100.

Los tres paneles comparten escala 0–100 %. Cada barrio se dibuja con el
resumen de su distrito, incluidos los barrios sin observaciones propias de
train. **No es una estimación por barrio**. Las líneas finas delimitan barrios;
no representan distintos valores dentro del distrito. Un distrito sin
observaciones se pintaría gris, no como 0 % de ninguna categoría.

Los números identifican distritos en la leyenda; sus posiciones se eligen
para rotular el dibujo, no son centros oficiales. Longitud y latitud están
en grados; el aspecto se corrige aproximadamente para Barcelona, sin una
proyección que permita medir distancias o superficies con precisión. Una
zona grande en el mapa no implica más anuncios: leer los tamaños `n`.

El archivo descargado tiene 75 objetos geométricos y 73 nombres únicos de
barrio. Vallvidrera, el Tibidabo i les Planes ocupa tres objetos: se dibujan
todas sus piezas, **sin multiplicar anuncios**. Entrenamiento contiene
67 barrios, todos con geometría; seis nombres del mapa no tienen filas de
train. La cifra 68 de la descripción original de la issue no se impone al
reparto actual ni se comprueba mirando test. Se informa el resultado observado.

El GeoJSON se descargó del enlace del README en `data/raw/`, sin reemplazar
un archivo anterior; Git lo ignora. Fuente: Inside Airbnb, Barcelona,
snapshot 2026-09-17, CC BY 4.0. La geometría original no forma parte de la
entrega. Su firma y la del CSV están en `eda11_metadata.json`.

## Modalidades, denominadores y grupos pequeños

`room_type` expresa modalidad alquilada, no tipo de propiedad. Se conservan
sus valores originales: Entire home/apt, Private room, Shared room y Hotel
room. Las traducciones son solo etiquetas de los gráficos.

Las barras presentan composición dentro de cada distrito o modalidad:
cada barra suma 100 %. No responden a qué porcentaje de todos los Premium
de Barcelona vive en un distrito concreto; esa pregunta usa otro denominador.

En el cruce, cada porcentaje se calcula dentro del grupo distrito × modalidad.
Las casillas en igual posición de los tres paneles suman 100 % si hay datos;
las filas de un panel no tienen por qué sumar 100 %. El CSV guarda recuentos
exactos; las casillas redondean porcentajes para lectura.

- `n_listings`: número de anuncios del grupo; no son personas, viviendas de
  toda la ciudad ni reservas.
- `count_*`: número de anuncios de cada categoría real.
- `pct_*`: porcentaje de esa categoría dentro del grupo completo.
- `small_group`: grupo observado con menos de 30 anuncios, marcado con †.
  Es una señal descriptiva elegida para leer con prudencia, no un umbral de
  significación ni una garantía de privacidad.
- `empty_group`: combinación sin observaciones. Tiene recuentos cero y
  porcentajes ausentes, no porcentajes de 0 %. Se dibuja gris con `n=0`.

No se eliminan grupos pequeños ni se presentan sus porcentajes como rankings
fiables. Las funciones rechazan índices de filas desalineados, etiquetas
reales desconocidas y conflictos entre nombres/distritos. La geografía
ausente o sin correspondencia detiene el mapa con explicación; no se oculta.

## Resultados observados con el archivo local

Población: **10.088 anuncios de entrenamiento**, 10 distritos, 67 barrios y
cuatro modalidades. Categorías reales: 3.366 Económico, 3.359 Medio y 3.363
Premium. Límites finales de train: 122,0 y 249,72333333333327 euros por noche;
no se redondean para etiquetar.

Ejemplos de lectura, calculados del snapshot (no cifras inventadas):

| Grupo | Anuncios del grupo | Categoría real descrita | Anuncios de esa categoría | Porcentaje dentro del grupo |
|---|---:|---|---:|---:|
| Eixample | 4.044 | Premium | 1.838 | 45,45 % |
| Nou Barris | 63 | Económico | 45 | 71,43 % |
| Habitación privada | 2.413 | Económico | 1.676 | 69,46 % |
| Habitación compartida | 90 | Económico | 85 | 94,44 % |
| Alojamiento entero | 7.527 | Premium | 3.085 | 40,99 % |
| Habitación de hotel | 58 | Premium | 26 | 44,83 % |

Los distritos tienen entre 63 y 4.044 anuncios: no son muestras equilibradas.
Hay 40 combinaciones distrito × modalidad: **siete sin observaciones** y
**13 observadas con menos de 30 anuncios**. Por ejemplo, el 100 % Económico
en una combinación con un solo anuncio no prueba un patrón estable del barrio
o del distrito. Comparar porcentajes exige leer siempre el denominador.

Las modalidades y los distritos difieren en composición de categorías reales.
El desglose ayuda a observar que también difiere la mezcla de modalidades;
no demuestra un efecto independiente o causal de ubicación o modalidad. No
se deduce utilidad predictiva, ingresos, ocupación ni rendimiento futuro.

## Salidas y reproducción

Se generan únicamente resultados agregados:

- `docs/results/eda11_districts.csv`: 10 filas.
- `docs/results/eda11_room_types.csv`: cuatro filas.
- `docs/results/eda11_district_room_types.csv`: 40 filas, incluidos vacíos.
- `docs/results/eda11_metadata.json`: población, umbrales, cobertura, tamaños,
  versiones y firmas de archivos/código, sin identificadores de anuncios.
- Cuatro figuras incrustadas en el notebook (mapas, dos barras y desglose).

No se llaman `*_comparison.csv`, para evitar mezclar descripciones con métricas
de modelos en la tabla de selección. El commit registrado es la base de la
ejecución; las firmas de código identifican también cambios locales aún no
publicados. Se verificó en Python 3.15.0 con las versiones registradas; no
se afirma haber repetido esta ejecución en Python 3.12 u otro sistema.

Requisitos: entorno del proyecto y herramientas de `requirements-notebooks.txt`,
CSV y GeoJSON del README. No hacen falta GeoPandas, Shapely, servicios de mapas,
claves de API, Docker o MongoDB. Desde la raíz:

```sh
source .venv/bin/activate
python -m pytest -q tests/test_eda_map.py tests/test_eda_map_results.py
python tools/run_eda_map_notebook.py
```

El primer comando de Python usa datos sintéticos y agregados guardados, sin
Airbnb raw ni GeoJSON; puede crear cachés. El segundo lee ambos archivos locales,
consume CPU y reemplaza el notebook y sus cuatro archivos agregados. Antes de
repetir, conservar una copia de esas cinco salidas si se quieren mantener.
No modifica los datos, no publica nada y no escribe en servicios externos.

También se puede ejecutar el notebook desde el editor con el kernel del
proyecto. No ejecutar ambas opciones simultáneamente. Si falta el GeoJSON,
seguir su enlace del README; no sustituirlo silenciosamente por otra ciudad
o versión. Ante un nombre sin correspondencia, revisar los archivos y
documentar la diferencia, no eliminar anuncios para forzar la cobertura.

Para verificación numerada y manejo de fallos, usar la sección de la issue 11
de `docs/pruebas_manuales.md`. Una ejecución fallida no queda validada por
encontrar archivos de resultados de una ejecución anterior.

## Límites

Es una descripción de anuncios históricos capturados y conservados tras la
limpieza, no de toda la oferta de vivienda de Barcelona ni de usuarios reales.
No se conocen aquí los precios finalmente pagados. El reparto aleatorio no
separa anfitriones ni tiempo. Los grupos provisionales del reparto se calculan
globalmente y los límites finales de clase se fijan con train completo, como
explica el protocolo común. Las asociaciones no son causas ni autorizan
cambiar la selección de variables o los modelos sin otro trabajo acordado.

La automatización de estados del Project todavía no está integrada. Esta
entrega local no mueve tarjetas, no publica cambios y no implica aprobación
ni merge. La documentación describe el análisis y sus comprobaciones, no un
acuerdo de publicación.

## Verificación realizada

- Notebook ejecutado completo: ocho celdas de código, sin salidas de error.
- Cuatro figuras incrustadas revisadas visualmente: mapas de las tres
  categorías, barras por distrito, barras por modalidad y cruce en tres
  paneles. Ejes, leyendas, números y notas legibles, sin recortes visibles
  en la revisión. No se ha probado impresión física ni todos los editores.
- 17 tests de esta entrega correctos: agregación, denominadores, grupos
  vacíos/pequeños, correspondencias, piezas geométricas repetidas, anillos,
  renderizado, rechazo de entradas incoherentes, protección del test reservado
  y coherencia de las salidas agregadas. No requieren los archivos raw.
- Suite completa: **84 tests correctos** con esta base. Incluye pruebas
  anteriores con entrenamientos pequeños sintéticos; el notebook EDA no
  entrena ningún modelo real ni usa el test reservado.
- `git diff --check` correcto.
- Ejecutado en memoria el lector común actual: sigue recogiendo cuatro CSV
  de modelos y siete filas; no incorpora los archivos `eda11_*`. No se
  modifica ese notebook ni se declara un modelo ganador.
