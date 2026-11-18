# 🏠 bcn-price-classifier

Clasificador del **segmento de precio anunciado por noche** de alojamientos
Airbnb en Barcelona: **Económico, Medio o Premium**. Proyecto educativo de
clasificación multiclase, con preparación compartida, comparación de modelos
y una aplicación Streamlit desplegada en Render con persistencia en MongoDB Atlas.

**[Abrir la demo pública](https://bcn-price-classifier.onrender.com)**

> Render Free puede tardar alrededor de un minuto en despertar tras inactividad.
> La demo guarda predicciones y respuestas: utiliza ejemplos ficticios, no datos
> personales. No necesitas instalar nada para abrirla.

## Contenido

- [Objetivo y límites](#objetivo-y-límites)
- [Demo y funcionamiento](#demo-y-funcionamiento)
- [Instalación local](#instalación-local)
- [Datos y atribución](#datos-y-atribución)
- [Preparación y evaluación](#preparación-y-evaluación)
- [Modelos y resultados registrados](#modelos-y-resultados-registrados)
- [Docker local](#docker-local)
- [Despliegue en Render y Atlas](#despliegue-en-render-y-atlas)
- [Seguimiento y datos nuevos](#seguimiento-y-datos-nuevos)
- [Pruebas y verificación](#pruebas-y-verificación)
- [Resolución de problemas](#resolución-de-problemas)
- [Estructura y documentación](#estructura-y-documentación)
- [Seguridad y licencias](#seguridad-y-licencias)

## Objetivo y límites

La app ofrece una orientación inicial del segmento al que se parece un anuncio
según sus características. **No recomienda un precio óptimo**, ni predice
ingresos, ocupación futura o el precio finalmente pagado. `Premium` significa
precio anunciado alto según los umbrales del modelo, no lujo ni calidad.
El beneficio económico para anfitriones es una hipótesis, no un efecto medido.

Las categorías se definen con los precios de entrenamiento. En el artefacto
distribuido se utilizan estos límites en euros por noche:

| Categoría real según precio | Regla exacta |
|---|---|
| Económico | Precio positivo ≤ 122,0 |
| Medio | 122,0 < precio ≤ 249,72333333333327 |
| Premium | Precio > 249,72333333333327 |

La interfaz redondea para mostrar los rangos, pero no para calcular categorías.
No son umbrales universales: corresponden al snapshot y al protocolo registrados.
Las probabilidades mostradas son estimaciones del modelo, no garantías de acierto.

## Demo y funcionamiento

1. Abre [la demo](https://bcn-price-classifier.onrender.com).
2. Completa barrio, tipo de alojamiento/habitación, capacidad, dormitorios,
   camas, baños, noches mínimas y servicios. La app deriva distrito y
   coordenadas de referencia por barrio; no solicita una dirección exacta.
3. Pulsa **🔮 Predecir**: muestra la categoría y probabilidades y trata de
   guardar una predicción. **Este botón escribe un registro en Atlas.**
4. Opcionalmente, introduce el precio anunciado real por noche y pulsa
   **Enviar**. **Escribe una respuesta**, denominada *feedback*, enlazada
   con la predicción. No cambia ni reentrena el modelo.

Para una prueba ficticia puedes usar **165 €**. Su categoría real según los
umbrales actuales es Medio; la categoría predicha depende de las características
introducidas. Solo hay acierto si la categoría predicha también es Medio.
El precio declarado no se verifica y las respuestas son voluntarias.

La demo carga **XGBoost champion**, versión `xgboost-trial34-6e9bc3e`, desde
[`models/app_model.joblib`](models/app_model.joblib). Está incluido en Git y
no necesita raw ni entrenamiento al arrancar. Su tamaño, SHA-256 y dependencias
figuran en el [manifiesto](models/app_model.manifest.json). Solo cargar
artefactos de procedencia confiable: `joblib.load` puede ejecutar código.

El archivo distribuido contiene únicamente el champion. Aunque el código
soporta A/B con LightGBM, **la demo actual no compara dos modelos**.

## Instalación local

Prerrequisitos: **Python 3.12**, Git y una terminal. Docker Desktop/Engine con
Compose solo es necesario para la vía Docker. Ejecuta los comandos desde la
raíz del repositorio. La rama de integración es `dev`.

### 1. Obtener el código y crear un entorno

Los siguientes comandos descargan el repositorio y crean archivos locales:

```bash
git clone --branch dev https://github.com/T6-P5-Classification/bcn-price-classifier.git
cd bcn-price-classifier
```

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows, PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si la política de PowerShell impide activar el entorno, se puede utilizar
directamente `.\.venv\Scripts\python.exe` en lugar de `python`, sin cambiar
políticas del equipo. En macOS/Linux el equivalente es `.venv/bin/python`.

### 2. Instalar dependencias

Elige **una** vía; la instalación descarga paquetes y modifica solo el entorno
virtual activado:

```bash
# Solo ejecutar la app con el modelo incluido:
python -m pip install -r requirements-app.txt

# O desarrollo, entrenamiento y tests:
python -m pip install -r requirements.txt
```

La imagen usa `xgboost-cpu`, con la misma versión de XGBoost que el proyecto.
No se necesita GPU. En macOS, si falta OpenMP para XGBoost/LightGBM, instalar
`brew install libomp` modifica el sistema local mediante Homebrew. En Linux,
OpenMP suele requerir el paquete `libgomp1`; el Dockerfile ya lo instala.

Para ejecutar notebooks mediante las herramientas de `tools/`, instalar además:

```bash
python -m pip install -r requirements-notebooks.txt
```

En VS Code/Jupyter, seleccionar el intérprete/kernel de `.venv`. Comprobar las
dependencias, sin entrenar ni consultar datos:

```bash
python -c "import xgboost, lightgbm, sklearn, streamlit, pymongo; print('OK')"
```

### 3. Abrir la app

```bash
python -m streamlit run app/app.py
```

Abre **http://localhost:8501**. Con las dependencias y el artefacto incluido,
la app no necesita descargar el dataset. Si `MONGODB_URI` no está configurada,
avisa de **CSV local** y guarda en `data/feedback/`. Si ya tienes esa variable
en tu entorno, utilizará su servidor: revisa la configuración antes de probar.
No uses credenciales o bases compartidas para pruebas sin autorización.

## Datos y atribución

- **Fuente:** [Inside Airbnb](https://insideairbnb.com/get-the-data/), Barcelona.
- **Snapshot de referencia:** **17 de septiembre de 2026**.
- **Licencia de los datos:** [Creative Commons Attribution 4.0 International
  (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- **Transformaciones del proyecto:** limpieza de precios/nulos, variables
  derivadas, selección de entradas, separación train/test y categorías por
  cuantiles. No se presenta el resultado como el dataset original sin cambios.

Los raw **no se suben a GitHub**. Solo necesitas descargarlos para explorar,
entrenar o ejecutar los controles que los requieren, no para abrir la demo:

| Archivo local | Uso | Descarga |
|---|---|---|
| `data/raw/listings.csv.gz` | Preparación, entrenamiento y control de procedencia | [listings](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/data/listings.csv.gz) |
| `data/raw/neighbourhoods.geojson` | Mapa de barrios | [neighbourhoods](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/visualisations/neighbourhoods.geojson) |

Descarga con el navegador y guarda con esos nombres. **No sobrescribas un
snapshot existente sin conservar una copia** si necesitas reproducirlo. Un
archivo de otra fecha no reproduce automáticamente los resultados registrados;
la procedencia del champion incluye el hash del raw utilizado.

Al reutilizar datos o resultados derivados, conserva la atribución a Inside
Airbnb, la fecha, el enlace a CC BY 4.0 e indica las transformaciones.
La licencia de los datos **no otorga por sí sola una licencia al código o al
modelo**; véase [Seguridad y licencias](#seguridad-y-licencias).

## Preparación y evaluación

La fuente ejecutable es [`src/cleaning.py`](src/cleaning.py) y
[`src/preprocessing.py`](src/preprocessing.py), con acuerdos en
[`docs/decisiones.md`](docs/decisiones.md).

- `load_train_test()` reutiliza limpieza, ingeniería de variables, selección,
  split **80/20** con semilla **42** y creación de categorías. El snapshot de
  referencia produce **10.088 anuncios de entrenamiento y 2.522 de test**.
- `MODEL_FEATURES` es una lista cerrada de **23 entradas antes del encoding**;
  `price`, sus derivados prohibidos e identificadores/datos personales
  excluidos no se usan como entradas. Una columna raw nueva no entra sola.
- Los extremos de precio positivos válidos no se eliminan automáticamente.
  Los precios inválidos/no positivos se excluyen porque no permiten crear
  la categoría real. Los nulos predictivos permitidos llegan al Pipeline.
- `build_preprocessor()` aplica mediana y escalado a numéricas; moda y
  one-hot a categóricas. Se ajusta **dentro del Pipeline y de cada fold**,
  nunca sobre test ni sobre todo train antes de validar.
- Validación: `StratifiedKFold(5, shuffle=True, random_state=42)`. Optuna y la
  selección se guían por **F1 macro medio de validación**, no por test.
  El gap es F1 de entrenamiento menos F1 de validación; el criterio registrado
  exige gap < 0,05. Un gap pequeño no garantiza utilidad ni ausencia de sesgo.
- Los umbrales finales se aprenden con train y se reutilizan en test. **Salvedad:**
  la estratificación usa bins provisionales calculados con todos los precios
  limpios; y los umbrales finales se fijan con todo train antes de los folds.
- Los diagnósticos **fuera de fold (OOF)** predicen cada fila de train con un
  modelo que no la vio en su ajuste, pero reutilizan la validación de selección:
  no son validación anidada ni una evaluación final independiente.
- Test queda reservado para evaluación final del champion, **ya registrada**.
  La baseline mostró históricamente resultados de esa partición; esa exposición
  está declarada y no se presenta independencia histórica perfecta.

Las 23 entradas se agrupan así; nombres exactos en `MODEL_FEATURES`:

| Grupo | Variables |
|---|---|
| Ubicación (4) | Barrio, distrito, latitud, longitud |
| Alojamiento (6) | Tipo de propiedad, habitación, huéspedes, dormitorios, camas, noches mínimas |
| Baños (3) | Cantidad, indicador compartido, indicador privado |
| Servicios (10) | Aire acondicionado, calefacción, secador de pelo, aparcamiento, self check-in, zona de trabajo, TV, piscina, ascensor, lavavajillas |

El modelo histórico se entrena con las coordenadas disponibles en el snapshot;
la app utiliza referencias por barrio. Es una limitación al interpretar las
predicciones, no una dirección exacta proporcionada por el usuario.

## Modelos y resultados registrados

Se comparan regresión logística, Random Forest, XGBoost y LightGBM con una
referencia Dummy. El champion registrado es **XGBoost, prueba 34 de Optuna**.
Su búsqueda tuvo **40 pruebas**; no se atribuye ese presupuesto a todos los
algoritmos. Tabla y protocolo: [comparación](docs/model_comparison.md).

Resultados leídos de [`champion_test.json`](docs/results/champion_test.json),
no recalculados para este README. Configuración y procedencia están en ese archivo:

| Partición y medida | Resultado |
|---|---|
| Validación cruzada, media F1 macro del champion | 0,7631, desviación estándar 0,0032 entre cinco folds |
| Test reservado, accuracy | 76,45 %: 1.928 aciertos sobre 2.522 anuncios |
| Test reservado, F1 macro | 0,7651 en escala 0–1, mismo peso para las tres clases |
| Test reservado, errores extremos | 31 sobre 2.522 anuncios: 1,23 % |

Un error extremo es **categoría real Económico y predicha Premium**, o
**categoría real Premium y predicha Económico**. En las matrices, las filas
representan la categoría real según precio/umbrales y las columnas la predicha.
Estos resultados no prueban rendimiento futuro, precio óptimo ni impacto
económico. La categoría Medio es la más difícil en el informe final registrado.

### Reproducir experimentos no es arrancar la app

Requiere el snapshot correcto, dependencias de desarrollo y kernel del proyecto.
No hace falta repetir búsquedas o test para revisar documentación o usar la demo.

| Recurso | Qué hace y dónde consultar efectos |
|---|---|
| `notebooks/04_model_logreg.ipynb` | Baseline y búsqueda; [guía](docs/logreg.md) |
| `notebooks/05_model_rf.ipynb` | Random Forest; código compartido en `src/rf_model.py` |
| `notebooks/06_model_xgboost.ipynb` | Búsqueda XGBoost; [ejecución y resultados](docs/xgboost.md) |
| `notebooks/07_model_lightgbm.ipynb` | LightGBM; resultados agregados en `docs/results/lightgbm_*` |
| `notebooks/08_comparison.ipynb` | Lector de comparación y sección separada de errores; [contrato](docs/model_comparison.md) y [análisis](docs/analisis_errores.md) |
| `notebooks/08b_champion_test.ipynb` | Evaluación final ya registrada; no repetir para elegir modelos ni ajustar decisiones |
| `notebooks/08c_feature_importance.ipynb` | Importancias del champion; no equivalen a efectos causales |

**Ejecutar todo el notebook de comparación repite ajustes OOF**, no solo lee
CSV. Las herramientas de `tools/` pueden reemplazar notebooks y resultados:
consulta su guía y conserva copias antes de repetir. No ejecutar toda la carpeta
de notebooks indiscriminadamente. El notebook MLP está vacío; no es un quinto
modelo entrenado o desplegado.

`python -m src.train_app_model` ajusta modelos con train y **reemplaza**
`models/app_model.joblib`; actualmente genera champion y challenger. No evalúa
test de nuevo, aunque lee métricas históricas verificadas. No ejecutarlo como
paso de instalación: cambiar el artefacto distribuido exige revisar contenido,
dependencias y actualizar el manifiesto y sus pruebas. La autorización para
abrir la app no autoriza publicar otro modelo.

## Docker local

### App sola, CSV efímero

Con Docker activo y sin necesidad de raw, estos comandos construyen una imagen
y crean un contenedor local. No entrenan ni publican la imagen:

```bash
docker build -t bcn-price-classifier .
docker run --rm --name bcn-price-demo -p 127.0.0.1:8501:8501 bcn-price-classifier
```

Abre **http://localhost:8501**. Sin URI configurada usa CSV en el contenedor:
**al eliminarlo se pierden esos registros**. `--rm` lo elimina al terminar;
usar solo ejemplos ficticios. No es la opción para conservar feedback.

### App y MongoDB persistente con Compose

```bash
docker compose up --build --wait
```

Descarga imágenes/dependencias si faltan, crea servicios y volúmenes y abre
la app en **http://localhost:8501**. El puerto debe estar libre. Compose pasa
`MONGODB_URI` apuntando al servicio `mongo`; la app guarda en MongoDB, no en Atlas.
MongoDB no publica puertos al exterior. Es una configuración local sin
autenticación, no una receta de producción.

- `mongo-data` conserva los registros MongoDB al recrear contenedores.
- `feedback-data` conserva CSV de ejecuciones anteriores; **no se migran**.
- `docker compose down` detiene/elimina los contenedores del entorno y conserva
  volúmenes. Interrumpe la app local: no hacerlo sobre un entorno compartido.
- **No usar `docker compose down -v` para parar:** elimina datos de volúmenes.

Pruebas y consulta de recuentos: [MongoDB](docs/mongodb.md).

## Despliegue en Render y Atlas

**Render ejecuta la app y carga el modelo; Atlas aloja MongoDB y conserva los
registros.** El Dockerfile despliega la app, no el Compose completo ni los
volúmenes de tu ordenador. No se transfiere el dataset de entrenamiento.

### Estado verificado

- URL: **https://bcn-price-classifier.onrender.com**.
- Render Free, Docker, Frankfurt; modelo `xgboost-trial34-6e9bc3e`.
  El despliegue inicial verificado utilizó **`657afa5`**. Posteriormente,
  el usuario cambió Render a **`dev`** y confirmó que la app funciona; no se
  ha verificado directamente el nuevo hash desplegado.
- Rama actual **`dev`**, según confirmación del usuario. Última configuración
  observada de **Auto-Deploy: Off**. El cambio remoto lo realizó el usuario,
  no esta edición documental.
- `main` sería una opción de
  entrega estable solo después de un acuerdo del equipo y de integrar allí
  la versión revisada. No fusionar ramas ni cambiar el servicio como efecto
  implícito de editar documentación.
- El usuario mostró una predicción y respuesta ficticias enlazadas en Atlas
  y confirmó que permanecían después de reiniciar la app. El agente comprobó
  HTTP 200 / `ok` en `/_stcore/health`, no consultó directamente Atlas.
- No se ha probado carga con varios usuarios ni recuperación de backups.

### Configuración para reproducir el despliegue

Crear servicios, guardar secretos o cambiar reglas de red modifica sistemas
remotos; no es un paso de revisión de solo lectura. Confirmar cuenta, cuotas y
plan antes de actuar; no modificar otros servicios del workspace.

1. **Atlas Free:** crear una base gratuita sin dataset de muestra. Crear un
   usuario de aplicación con `readWrite` solo en `bcn_price_classifier` y
   restringirlo al clúster de la demo; no utilizar el administrador.
2. **Render → New → Web Service:** seleccionar el repositorio y una rama remota
   que ya contenga el modelo y Dockerfile. Para un despliegue nuevo desde código
   integrado, usar `dev`. Las ramas de trabajo sirven para pruebas previas.
3. Revisar los campos:

   | Campo | Valor |
   |---|---|
   | Language | Docker |
   | Region | Frankfurt en la demo actual |
   | Instance Type | Free, $0/mes dentro de los límites del plan |
   | Docker Build Context Directory | `.` (un punto) |
   | Dockerfile Path | `Dockerfile` |
   | Health Check Path | `/_stcore/health` |
   | Docker Command / Pre-Deploy Command | Vacíos; usar el CMD incluido, sin entrenamiento |
   | Auto-Deploy | Off para despliegue manual |
   | Registry Credential / Secret Files | No necesarios para esta configuración |

4. En **Environment Variables**, configurar **Key = `MONGODB_URI`** y
   **Value = la conexión completa de Atlas**. Empieza por `mongodb+srv://` e
   incluye usuario, contraseña y `/bcn_price_classifier` antes de `?` si hay
   opciones. Sustituir los marcadores de plantilla completos, incluidos `< >`.
   El `@` que separa contraseña y servidor sí se conserva. Codificar caracteres
   reservados de la contraseña; no copiar la contraseña del administrador.
5. No hacen falta variables adicionales `PORT` ni `FEEDBACK_DIR`: el Dockerfile
   respeta el puerto de Render y la app debe guardar en Atlas. **Nunca pegar
   secretos en Git, capturas, chat ni comandos que queden en el historial.**
6. **Deploy web service** envía código/modelo a Render, consume cuota e inicia
   un servicio con URL pública. Tras crearlo, consultar **Connect → Outbound
   IP Addresses** y añadir esos rangos concretos a **Atlas → IP Access List**.
   Son rangos compartidos con otros servicios; la autenticación sigue siendo
   obligatoria. La IP local no autoriza a Render. No abrir `0.0.0.0/0`.
7. Esperar a entradas de red Active y despliegue Live. Abrir el formulario,
   probar predicción y respuesta ficticias y comprobar los registros enlazados
   en Atlas. Con permiso para interrumpir la app, reiniciar solo el servicio
   Render y comprobar que ambos registros permanecen, sin crear otros nuevos.

**Persistente no significa gratuito ilimitado ni copia de seguridad.** Render
Free puede suspenderse por inactividad y perder archivos locales; el historial
está en Atlas, no en esos archivos. Atlas Free tiene límites y políticas de
inactividad propios. No activar planes o copias de pago sin autorización.

Guía de operación, requisitos, evidencias y verificación detallada:
**[docs/render.md](docs/render.md)**. Un HTTP 200 no prueba el guardado en Atlas.

### Variables y selección del almacenamiento

| Variable / argumento | Función |
|---|---|
| `MONGODB_URI` | Selecciona MongoDB si no se pasa `directory=` explícito; valor secreto, no una clave API |
| `FEEDBACK_DIR` | Carpeta CSV, por defecto `data/feedback/`; no selecciona CSV si existe URI |
| `APP_MODEL_PATH` | Ruta alternativa a un artefacto confiable; por defecto `models/app_model.joblib` |
| `AB_SPLIT` | Fracción de sesiones para challenger cuando existe en el artefacto; sin challenger no tiene efecto |
| `PORT` | Puerto del contenedor, indicado por Render; 8501 por defecto en Docker local |
| `TEST_MONGODB_URI` | Solo integración automatizada con MongoDB local aislado; nunca Atlas |
| `directory=` | Argumento de las funciones de almacenamiento: selecciona CSV explícitamente para llamadas locales aisladas |

Crear un `.env` no basta para Streamlit: la app no llama automáticamente a
`load_dotenv()`. En Render se usan sus Environment Variables; localmente hay
que definir el entorno del proceso de forma segura. No se proporcionan secretos
reales en este repositorio.

Si MongoDB está configurado pero falla, **no se cambia silenciosamente a CSV**.
La predicción puede mostrarse, pero la app avisa del guardado fallido y no
permite enviar una respuesta para esa predicción no guardada. CSV y MongoDB
no se unen, migran ni borran automáticamente.

## Seguimiento y datos nuevos

- **Métricas de feedback:** en la app, abrir **Seguimiento del feedback** y
  activar **Mostrar métricas del feedback**. Lee agregados; no crea predicciones
  ni respuestas. Se actualiza cada diez segundos y con **Actualizar ahora**.
  Separa modelo, versión, variante y umbrales. Las métricas solo usan respuestas
  voluntarias, no representan a todos los anuncios. [Detalle](docs/feedback_metrics.md).
- **Datos nuevos:** `src/new_data.py` usa `load_labelled()` de la interfaz
  compartida. Valida entradas/precios y prepara características y categoría o
  solo características y precio con `include_target=False`. **No reentrena**.
- **A/B:** requiere regenerar y revisar un artefacto con champion y challenger.
  Solo entonces `python -m src.ab_test` lee el almacenamiento configurado y
  compara una respuesta por sesión, con al menos 100 sesiones por modelo y un
  intervalo de confianza del 95 % para declarar una diferencia. Con el modelo
  distribuido no ejecutar esa comparación: no existe challenger.
- **Data drift:** compara las 23 variables de un snapshot nuevo con train,
  no precios ni test. Requiere raw de referencia y snapshot nuevo. Ejemplo:

  ```bash
  python -m src.drift data/raw/listings_2026-06-24.csv.gz --no-save
  ```

  `--no-save` imprime sin reemplazar resultados; sin esa opción escribe CSV y
  JSON en `docs/results/`. El archivo de junio no se incluye: hay que obtenerlo
  aparte. El análisis ya registrado usa junio frente a septiembre y no acredita
  vigilancia de datos futuros. [Decisiones](docs/decisiones.md).
- **Reentrenamiento automático:** `src/retrain.py` está vacío; no está operativo.
  Conservar feedback no desencadena entrenamiento ni sustitución del modelo.

Contrato único de almacenamiento y consumidores:
[docs/feedback.md](docs/feedback.md).

## Pruebas y verificación

Usa el entorno de **desarrollo** con `requirements.txt`. Los tests pueden
crear cachés y archivos sintéticos temporales; no implican permiso de publicación,
aprobación, merge o ejecución de pruebas contra bases compartidas.

```bash
# Controles rápidos de artefacto y configuración Docker, sin raw ni Atlas:
python -m pytest -q tests/test_deployment_artifact.py tests/test_docker.py

# Integridad de preparación con datos sintéticos:
python -m pytest -q tests/test_data.py

# Suite completa (requiere snapshot raw local para controles de procedencia):
python -m pytest -q
```

Los tests de integridad comprueban precios válidos, separación/cobertura de
filas, asociación X/y, lista de entradas, nulos permitidos, aprendizaje de
umbrales, clases, estratificación, reproducibilidad, límites y datos no
separables. La app se prueba con artefactos sintéticos; otros controles revisan
el artefacto champion real y resultados agregados. No confundirlos con una
nueva evaluación de rendimiento del test reservado.

Sin `TEST_MONGODB_URI`, los **ocho tests de integración MongoDB se omiten**;
no describir esa ejecución como integración real. Para ejecutarlos, usar un
servidor **local aislado**, no Atlas: crean y eliminan bases aleatorias con
datos ficticios, nunca deben apuntar a datos de otra persona.
Procedimiento: [docs/mongodb.md](docs/mongodb.md).

Referencia de validación del despliegue: **358 tests correctos**, incluidos
los de MongoDB local, con tres avisos de gráficos no interactivos. Es un
resultado registrado, no una garantía para cualquier entorno o revisión.

### Verificación documental de esta entrega

En la rama `docs/41-readme-current-deployment` (base `dev` en `24e443b`):

1. **Solo lectura:** comprobar enlaces y que instalación, datos, uso, modelo,
   evaluación, almacenamiento y despliegue están identificados. No hay secretos
   ni comandos de entrenamiento presentados como requisito para abrir la app.
2. **Local:** ejecutar los controles rápidos anteriores. Esperado: nueve tests
   correctos; si faltan dependencias, preparar el entorno, no alterar el modelo.
3. **Solo lectura remota:** abrir la demo o su endpoint `/_stcore/health`.
   Esperado: formulario o HTTP 200 / `ok`; si está suspendida, esperar al arranque.
   No pulsar Predecir para una comprobación de solo lectura.
4. **Prueba con escritura, opcional y autorizada:** seguir el ejemplo de 165 €
   y verificación de Atlas en [la guía](docs/render.md). Requiere acceso propio
   autorizado a Atlas; no se entregan credenciales como requisito de revisión.
5. Ante fallo, distinguir entorno incompleto, enlace documental incorrecto y
   defecto de la app. No borrar datos, regenerar artefactos, desactivar tests
   ni ampliar permisos para conseguir un resultado verde.

Más procedimientos: [pruebas manuales](docs/pruebas_manuales.md) y
[contrato del modelo](docs/test_model.md), cuyas secciones históricas deben
interpretarse junto con el estado actual descrito aquí.

## Resolución de problemas

| Síntoma | Primera comprobación segura |
|---|---|
| La demo tarda en abrir | Render Free puede estar despertando; esperar antes de repetir |
| No encuentra el modelo | Revisar rama/checkout y `APP_MODEL_PATH`; el archivo distribuido debe estar presente |
| Aparece CSV local | Falta `MONGODB_URI` en el proceso; no basta crear `.env` |
| Predice pero no guarda | Revisar URI completa, usuario de app, contraseña codificada, base, IP de salida y errores sanitizados; no es necesariamente un fallo del modelo |
| Atlas rechaza autenticación | No usar `< >` de plantilla, contraseña de cuenta Atlas ni la de otro usuario; no conceder permisos de administrador para arreglarlo |
| Docker falla en COPY del modelo | La rama remota debe contenerlo; el archivo local de otra máquina no llega a Render |
| Fallo OpenMP en macOS | Revisar `libomp` y arquitectura/intérprete; no cambiar versiones a ciegas |
| Suite falla por raw ausente | Añadir el snapshot correcto al entorno local, no a Git |
| Integración MongoDB sale skipped | Falta `TEST_MONGODB_URI` local; no equivale a conexión probada |
| CLI A/B dice que no hay challenger | Limitación deliberada del artefacto distribuido; no regenerarlo automáticamente |

Nunca publicar la URI, contraseñas ni documentos reales para pedir ayuda.
Para investigar, compartir solo mensajes sanitizados y configuración no secreta.

## Estructura y documentación

```text
app/          Streamlit, formulario, predicción y panel de feedback
src/          Preparación, modelos, almacenamiento, métricas y drift
notebooks/    Preparación (01a–01e), EDA (02–03), modelos (04–07), análisis (08*)
models/       Champion distribuido y manifiesto; otros artefactos no por defecto
data/raw/     Snapshot local y mapa; ignorados por Git
data/feedback/ CSV locales; ignorados por Git
tests/        Pruebas unitarias, sintéticas, de artefacto e integración opt-in
tools/        Ejecutores de notebooks; revisar efectos antes de utilizarlos
docs/         Decisiones, guías y resultados agregados
```

| Tema | Referencia |
|---|---|
| Acuerdos y salvedades de modelado | [Decisiones](docs/decisiones.md) |
| Guía de comprobación por issue | [Pruebas manuales](docs/pruebas_manuales.md) |
| EDA por distrito y habitación | [EDA](docs/eda_distritos_room_type.md) |
| Regresión logística / XGBoost | [LogReg](docs/logreg.md) / [XGBoost](docs/xgboost.md) |
| Comparación y errores | [Comparación](docs/model_comparison.md) / [Errores](docs/analisis_errores.md) |
| Contrato del Pipeline | [Pruebas del modelo](docs/test_model.md) |
| Predicciones y feedback / MongoDB | [Contrato](docs/feedback.md) / [MongoDB](docs/mongodb.md) |
| Métricas del feedback | [Panel y denominadores](docs/feedback_metrics.md) |
| Render, Atlas y persistencia | [Despliegue](docs/render.md) |

Algunas guías conservan el contexto histórico de su entrega. Ante discrepancias,
contrastar con el código y las decisiones vigentes; no asumir que todos los
archivos vacíos o propuestas pendientes son funcionalidades disponibles.

## Seguridad y licencias

- No publicar raw, datos personales, contraseñas, `.env` reales ni historial
  operativo. El formulario guarda características y una sesión anónima, no
  nombres de usuarios. No asumir que esto sustituye una revisión de privacidad.
- Las coordenadas que registra la app son referencias agregadas por barrio.
  El entrenamiento histórico utiliza las coordenadas del snapshot.
- El champion y sus metadatos sí se distribuyen expresamente; esto no implica
  que se publique el dataset ni garantiza ausencia de riesgos de privacidad
  derivados del modelo aprendido.
- La licencia **CC BY 4.0 corresponde a los datos de Inside Airbnb**. No hay
  archivo `LICENSE` para el código en esta revisión: no se declara una licencia
  de software o del artefacto por inferencia. Un repositorio público o un fork
  no sustituyen una licencia explícita; acordarla es una tarea separada.
- El proyecto no modifica accesos, servicios, costes ni configuración de
  repositorios al ejecutar sus tests locales. Publicación y operaciones remotas
  requieren autorización y revisión independientes.
