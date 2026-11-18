# 🏠 bcn-price-classifier

Clasificación multiclase del segmento de precio (**Económico / Medio / Premium**) de los anuncios de Airbnb en Barcelona.

> ¿Podemos predecir el segmento de precio de un anuncio de Airbnb en Barcelona a partir de sus características?

## 📊 Datos

- **Fuente:** [Inside Airbnb](https://insideairbnb.com/get-the-data/), Barcelona, snapshot del 17/09/2026
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Los datos **no se suben a GitHub**. Descárgalos y guárdalos en `data/raw/`:

| Archivo | Para qué | Enlace |
|---|---|---|
| `listings.csv.gz` | Datos principales (EDA y modelos) | [descargar](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/data/listings.csv.gz) |
| `neighbourhoods.geojson` | Solo para dibujar el mapa de barrios | [descargar](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/visualisations/neighbourhoods.geojson) |

## ⚙️ Instalación

**1. Clonar el repositorio**
```bash
git clone https://github.com/T6-P5-Classification/bcn-price-classifier.git
cd bcn-price-classifier
```

**2. Crear y activar el entorno virtual** (Python 3.12)
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

**3. Instalar las dependencias**
```bash
pip install -r requirements.txt
```

**4. Solo en macOS:** XGBoost y LightGBM necesitan OpenMP
```bash
brew install libomp
```

**5. Comprobar la instalación**
```bash
python -c "import xgboost, lightgbm, sklearn; print('OK')"
```

> En VS Code / Jupyter, selecciona el kernel de `.venv` para los notebooks.

## 🧪 Tests de integridad de datos (#30)

Los controles de [`tests/test_data.py`](tests/test_data.py) comprueban que la
preparación entrega datos consistentes al entrenamiento. **No entrenan un
modelo ni miden su capacidad para predecir.** Complementan los tests de
limpieza y preprocesamiento existentes.

### Qué comprobamos

| Comprobación | Qué problema detecta |
|---|---|
| Precios válidos | Se excluyen filas sin precio válido, con precio cero, negativo o infinito; se conserva un precio positivo extremo. |
| Separación y cobertura | Cada fila válida aparece una sola vez, en entrenamiento o test; se respeta el reparto 80/20 con el redondeo necesario. |
| Asociación de filas | Las características y la categoría siguen correspondiendo al mismo anuncio. |
| Variables de entrada | Train y test tienen las columnas de `MODEL_FEATURES`, en su orden y con los mismos tipos; no entran precio, identificadores ni los campos excluidos comprobados. |
| Nulos permitidos | Los valores vacíos de `minimum_nights` se conservan; no se rellenan antes del preprocesamiento ni se exige un indicador descartado del modelo. |
| Aprendizaje de umbrales | Los límites de las categorías se calculan una sola vez con precios de entrenamiento, no con test ni con todo el dataset. |
| Categorías finales | Train y test usan los mismos límites de entrenamiento y las etiquetas ordenadas `Económico`, `Medio`, `Premium`, sin etiquetas nulas. |
| Estratificación | Los grupos provisionales de precio mantienen aproximadamente sus proporciones al separar test. Estos grupos no son las categorías finales. |
| Reproducibilidad y retorno | Repetir la llamada por defecto produce los mismos datos y mantiene el retorno habitual de cuatro objetos. |
| Precios en los límites | Un precio igual al límite inferior es `Económico`; uno igual al superior es `Medio`. |
| Datos no separables | Si todos los precios son iguales, se informa de un error en lugar de continuar sin poder formar tres grupos. |

### Cómo ejecutarlos

Desde la **raíz del repositorio**, con el entorno virtual activado y las
dependencias instaladas:

```bash
# Solo los controles de integridad de datos de #30
python -m pytest -q tests/test_data.py

# Todos los tests de Python del proyecto
python -m pytest -q
```

Los tests de #30 crean anuncios ficticios en un CSV comprimido temporal y
ejecutan la función real `load_train_test()`. **No necesitan descargar el
dataset de Airbnb**, acceder a una base de datos ni utilizar credenciales.
La tabla ficticia incluye las entradas necesarias para las 23 variables
acordadas. El esquema esperado se toma de `MODEL_FEATURES`, no de una lista
antigua de variables. No se fija como requisito el número de filas del
snapshot real. Las particiones de estos tests son sintéticas: no se consulta
ni se evalúa el test reservado del dataset de Airbnb.

### Cómo interpretar el resultado

- **`passed`**: el control se cumple para los casos de prueba definidos.
- **`FAILED`**: revisar el nombre del test y la comparación mostrada. Puede
  haber una regresión en la preparación o una expectativa que deba revisarse
  tras un cambio acordado; no borrar el test solo para que pase.
- Un valor vacío en una característica no siempre es un error: #13 aprende
  la imputación dentro del Pipeline usando solo entrenamiento.
- Que estos tests pasen no certifica la calidad de cualquier archivo nuevo
  de Airbnb ni el rendimiento del modelo. Las métricas y el funcionamiento
  del modelo tienen entregas separadas (#31 y #32).

Las reglas de preparación se documentan en
[`docs/decisiones.md`](docs/decisiones.md); estos tests verifican parte de
ese contrato. Esta sección documenta únicamente #30, no sustituye la
actualización completa del README prevista en #41.

### Modelo XGBoost

El notebook candidato usa 40 pruebas de Optuna y diagnósticos fuera de fold
solo sobre train: [ejecución y resultados](docs/xgboost.md). La evaluación
final posterior del champion ya está registrada en
[champion_test.json](docs/results/champion_test.json); no se repite al
ejecutar el candidato.
La guía incluye efectos locales y [pasos de verificación](docs/pruebas_manuales.md#issue-20--xgboost-con-40-pruebas-de-optuna).

### App Streamlit (#15)

Desde la raíz, con el entorno activo:

```bash
streamlit run app/app.py
```

La app usa el **champion XGBoost** (#22): los hiperparámetros de la prueba
de Optuna seleccionada, ajustado con todo train (no usa test). Si existe
`docs/results/champion_test.json`, la app muestra también su resultado en
test. Los tests de la app (`tests/test_app.py`) usan un modelo sintético y no
necesitan el snapshot.

La demo distribuye el champion revisado en `models/app_model.joblib`: no
requiere datos raw ni entrenamiento para abrirla. Procedencia, integridad y
preparación de Render con Atlas: [docs/render.md](docs/render.md).
**Demo pública:** [bcn-price-classifier.onrender.com](https://bcn-price-classifier.onrender.com).
Render Free puede tardar en despertar tras inactividad. Las predicciones y
respuestas se guardan en Atlas; usar entradas ficticias para probar la demo.

Cada predicción y, si el usuario lo indica, su precio real por noche se
guardan en MongoDB cuando se configura `MONGODB_URI`. Sin esa configuración
se usan CSV en `data/feedback/` (fuera de Git), con un aviso visible en la app.
No hay migración automática ni cambio a CSV si MongoDB falla. Contrato:
[`docs/feedback.md`](docs/feedback.md); configuración y comprobación:
[`docs/mongodb.md`](docs/mongodb.md).

### Data drift (#36)

Compara las 23 variables de un snapshot nuevo de Inside Airbnb con las filas
de **train** con las que se entrenó el modelo (PSI por variable; no usa
precios ni test):

```bash
python -m src.drift data/raw/listings_2026-06-24.csv.gz
```

Resultado en `docs/results/drift_<fecha>.csv` y `.json`. Resultados y
lectura en [`docs/decisiones.md`](docs/decisiones.md#data-drift-snapshot-de-junio-frente-a-train-36).

### A/B testing (#35)

Requiere un artefacto con ambos modelos. El distribuido para la demo (#33)
contiene solo el champion: sirve XGBoost a todas las sesiones y no permite
comparar variantes. El comando `python -m src.train_app_model` genera ambos
modelos usando el snapshot; sustituir el archivo distribuido requiere revisión.

Cada sesión de la app recibe al azar el **champion** (XGBoost) o el
**challenger** (LightGBM, empate práctico en CV), y la predicción se guarda
con su `variant` y un identificador anónimo de sesión. La proporción del challenger se cambia con `AB_SPLIT`
(por defecto `0.5`; `0` desactiva el test). Para comparar los aciertos con
el feedback recibido:

```bash
python -m src.ab_test
```

Solo cuenta las respuestas de los dos modelos del artefacto actual (modelo y
versión) y **una por sesión**. Declara un ganador solo con al menos 100
sesiones por modelo y una diferencia clara (intervalo del 95 %). Detalles en
[`docs/decisiones.md`](docs/decisiones.md#ab-testing-champion-frente-a-challenger-35).

### Docker (#29)

El champion revisado está incluido: construir y arrancar no entrenan modelos.

```bash
docker build -t bcn-price-classifier .
docker run -p 8501:8501 bcn-price-classifier   # http://localhost:8501
```

La imagen usa `requirements-app.txt` (solo lo que necesita la app, mismas
versiones) y `.dockerignore` deja fuera datos, `.venv`, notebooks y docs. El
puerto se puede cambiar con la variable `PORT` (p. ej. en Render).

### Docker Compose: app + base de datos (#26)

```bash
docker compose up --build       # app en http://localhost:8501
docker compose down             # para; los datos se conservan
```

Arranca la app y MongoDB. Compose pasa `MONGODB_URI` a la app, que guarda en
las colecciones `predictions` y `feedback`; el volumen `mongo-data` conserva
los datos al parar o recrear contenedores. Los CSV anteriores del volumen
`feedback-data` siguen intactos, pero no se migran automáticamente.
`docker compose down -v` borra los datos de esos volúmenes: no usarlo para
una parada normal. La configuración es local; no configura Atlas (#28).

## 🗂️ Estructura

```
data/        Datos (no se suben a GitHub)
notebooks/   Exploración, EDA y modelos (01a–01d, 02–09)
src/         Código reutilizable (limpieza, preprocesamiento, BD…)
app/         Aplicación Streamlit
models/      Modelos entrenados
tests/       Tests con pytest
docs/        Decisiones e informe técnico
```

## Informe técnico

[Problema, datos, evaluación y resultados del proyecto](docs/informe_tecnico.md).
### Métricas del feedback (#25)

En la app, abrir **Seguimiento del feedback** y activar **Mostrar métricas del
feedback**. El panel muestra agregados por modelo, versión, variante y umbrales,
con actualización cada diez segundos y manual. Sin respuestas no presenta
scores. [Uso, denominadores, pruebas y límites](docs/feedback_metrics.md).
