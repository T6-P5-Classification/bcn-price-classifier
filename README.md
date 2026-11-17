# bcn-price-classifier

Clasificación del segmento de **precio anunciado por noche** de alojamientos
Airbnb en Barcelona: **Económico, Medio y Premium**.

El objetivo es orientar al anfitrión sobre el segmento de su anuncio a partir
de sus características. Premium significa precio alto respecto a los umbrales
del snapshot, no necesariamente lujo. El proyecto no estima un precio óptimo,
ingresos, ocupación futura ni el precio finalmente pagado.

## Estado y requisitos

La rama de integración es `dev`. Están disponibles la limpieza, el feature
engineering, la selección de 23 entradas, el split y el preprocesamiento
compartidos. Los notebooks de modelos, el champion y la aplicación se integran
por entregas: un archivo presente pero vacío no constituye una implementación.

Requisitos locales: Git, **Python 3.12**, acceso al dataset y espacio para un
entorno virtual. En macOS, XGBoost y LightGBM necesitan OpenMP (`libomp`).
No hacen falta Docker ni MongoDB para ejecutar la preparación y los modelos.

## Instalación

Clonar la rama de integración:

```bash
git clone --branch dev https://github.com/T6-P5-Classification/bcn-price-classifier.git
cd bcn-price-classifier
```

Crear el entorno en macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si no se puede activar el entorno, invocar directamente su Python:
`.venv/bin/python` en macOS/Linux o `.\.venv\Scripts\python.exe` en Windows.
No es necesario cambiar la política de seguridad de PowerShell.

Solo en macOS, instalar OpenMP con Homebrew antes de comprobar los modelos:

```bash
brew install libomp
```

Con el entorno activo, instalar las versiones del proyecto y comprobarlo:

```bash
python --version
python -m pip install -r requirements.txt
python -c "import xgboost, lightgbm, sklearn; print('OK')"
```

`python --version` debe indicar 3.12.x. En VS Code, instalar las extensiones
Python y Jupyter de Microsoft, abrir el repositorio y seleccionar
**Select Kernel → Python Environments → .venv** en cada notebook.
Las extensiones del editor son distintas de las dependencias de Python.

## Datos, descarga y licencia

- Fuente: [Inside Airbnb](https://insideairbnb.com/get-the-data/).
- Ciudad y snapshot: Barcelona, **17 de septiembre de 2026**.
- Licencia del dataset: [Creative Commons Attribution 4.0 International
  (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
- Atribución para resultados derivados: «Datos de Inside Airbnb, Barcelona,
  snapshot 2026-09-17, CC BY 4.0; limpieza y transformaciones propias».

La licencia citada corresponde a los datos de origen; no establece por sí
misma una licencia para el código del repositorio. Para reutilizar los datos,
conservar la atribución, los enlaces a fuente/licencia e indicar cambios.

Descargar los archivos sin descomprimir ni renombrar el principal:

| Archivo | Destino | Uso | Descarga |
|---|---|---|---|
| `listings.csv.gz` | `data/raw/` | Preparación, EDA y entrenamiento | [CSV comprimido](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/data/listings.csv.gz) |
| `neighbourhoods.geojson` | `data/raw/` | Visualización del mapa; no es necesario para el clasificador | [GeoJSON](https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/visualisations/neighbourhoods.geojson) |

Se pueden descargar desde esos enlaces con el navegador. **Atención:** el siguiente
comando reemplaza `data/raw/listings.csv.gz` si ya existe. Conservar una copia
del archivo actual antes de repetir la descarga. En macOS/Linux:

```bash
mkdir -p data/raw
curl -L --fail --show-error 'https://data.insideairbnb.com/spain/catalonia/barcelona/2026-09-17/data/listings.csv.gz' -o data/raw/listings.csv.gz
```

Comprobar que el archivo es legible, sin mostrar filas ni datos personales:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/raw/listings.csv.gz', compression='gzip', nrows=1).shape)"
git check-ignore data/raw/listings.csv.gz
```

`data/raw/` y `data/processed/` están excluidos de Git salvo sus `.gitkeep`.
No subir datasets, credenciales, archivos `.env` con valores reales ni salidas
que expongan información personal. Una descarga HTML o incompleta produce
un error al leer gzip: revisar el enlace y repetir la descarga.

## Uso: preparación y entrenamiento

Ejecutar los notebooks de preparación desde la carpeta `notebooks/`, con el
kernel de `.venv`, en este orden:

1. `01a_eda_initial.ipynb`: exploración inicial.
2. `01b_cleaning.ipynb`: limpieza compartida.
3. `01c_column_selection.ipynb`: separación de entradas y vistas.
4. `01d_feature_engineering.ipynb`: variables derivadas.
5. `01e_split_target.ipynb`: demostración del split y creación del target.
6. `02_eda_classes.ipynb`: análisis de clases y selección de variables.

Los notebooks de preparación documentan las etapas. **Un notebook de modelo
no necesita ejecutar los anteriores**: llama a las funciones compartidas.
El notebook del mapa requiere además el GeoJSON.

Desde la raíz del repositorio, comprobar el contrato común:

```bash
python -c "from src.preprocessing import load_train_test; a,b,c,d = load_train_test(); print(a.shape, b.shape); assert a.shape[1] == b.shape[1] == 23"
```

En los notebooks de modelos, con `notebooks/` como directorio de trabajo:

```python
import sys
sys.path.insert(0, '..')
from sklearn.pipeline import Pipeline
from src.preprocessing import load_train_test, build_preprocessor

X_train, X_test, y_train, y_test = load_train_test()
# estimador es el clasificador del notebook correspondiente.
# pipeline = Pipeline([('prep', build_preprocessor()), ('model', estimador)])
```

No repetir la lectura, limpieza, selección, split o creación de clases en cada
modelo. La ruta por defecto del CSV se resuelve desde el módulo, no desde el
directorio de trabajo.

Para ejecutar los entrenamientos disponibles, seguir las guías específicas:

- [Regresión logística: ejecución y resultados OOF](docs/logreg.md).
- [XGBoost: ejecución y resultados OOF](docs/xgboost.md).
- [Contrato y uso de la tabla común](docs/model_comparison.md).

El fragmento anterior ilustra la carga compartida; no ejecuta por sí solo un
entrenamiento. Cada guía indica los comandos y dependencias necesarios.

### Entradas y clases

El modelo usa la lista cerrada `MODEL_FEATURES` de `src/cleaning.py`:

| Grupo | Entradas |
|---|---|
| Localización (4) | `neighbourhood_cleansed`, `neighbourhood_group_cleansed`, `latitude`, `longitude` |
| Alojamiento y estancia (6) | `property_type`, `room_type`, `accommodates`, `bedrooms`, `beds`, `minimum_nights` |
| Baños (3) | `bathrooms_count`, `is_shared_bathroom`, `is_private_bathroom` |
| Servicios (10) | `has_air_conditioning`, `has_heating`, `has_hair_dryer`, `has_parking`, `has_self_checkin`, `has_dedicated_workspace`, `has_tv`, `has_pool`, `has_elevator`, `has_dishwasher` |

`price`, identificadores, datos personales e historial del anuncio excluido
no son entradas del modelo. Las 23 variables se cuentan **antes** del encoding;
el número de columnas transformadas puede variar entre folds.

El split compartido es 80/20, `random_state=42`. Los cuantiles 1/3 y 2/3 de
los precios de **train** definen las clases y se aplican sin recalcular a test:

- Económico: `price <= lower`.
- Medio: `lower < price <= upper`.
- Premium: `price > upper`.

No fijar los umbrales a mano ni redondearlos para etiquetar. Los bins
provisionales globales solo se usan para estratificar la partición; esta
salvedad y la definición de clases se explican en
[las decisiones comunes](docs/decisiones.md).

### Validación y comparación

- Usar el Pipeline completo en `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Aprender imputación, escalado y encoding dentro de cada fold de entrenamiento.
- Optimizar con Optuna únicamente en train, por F1 macro medio de validación.
- Registrar presupuesto, semillas, parámetros, versiones y tiempo; el presupuesto
  común de Optuna sigue pendiente de acuerdo.
- Reportar media/desviación de F1 macro de validación, media de train y
  `gap = media_train - media_validación < 0.05` absoluto (cinco puntos porcentuales).
- Comparar con DummyClassifier bajo el mismo protocolo. Las métricas por
  categoría y las matrices de cada modelo candidato se calculan con predicciones
  fuera de fold sobre entrenamiento. El test reservado se utiliza únicamente
  para la evaluación final del champion, es decir, el modelo seleccionado.
- En esa evaluación final, reportar accuracy, F1 macro, precision/recall/F1
  por clase y matriz de confusión.
- Orden común: Económico, Medio, Premium. En la matriz, filas = clase real según
  precio/umbrales; columnas = clase predicha.

El split aleatorio no separa anfitriones ni tiempo. La CV optimizada puede ser
optimista y una pequeña diferencia train–validación no garantiza buen rendimiento.
No usar test para escoger modelos, variables o hiperparámetros. Una evaluación
repetida sobre el mismo test no crea una nueva evaluación independiente.

## Pruebas

Desde la raíz, con el entorno activo:

```bash
python -m pytest -q
```

Los tests actuales de preparación usan datos sintéticos: no requieren descargar
el snapshot. Comprueban comportamiento del código, no calidad predictiva real.
Las pruebas con datos o modelos externos deben indicar sus requisitos y no
presentar un skip como una validación superada.

Para verificaciones paso a paso, consultar
[la guía de pruebas manuales](docs/pruebas_manuales.md).

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

Entrenamiento con 40 pruebas de Optuna, diagnósticos fuera de fold sobre
train y test reservado sin evaluar: [ejecución y resultados](docs/xgboost.md).
La guía incluye efectos locales y [pasos de verificación](docs/pruebas_manuales.md#issue-20--xgboost-con-40-pruebas-de-optuna).


## Aplicación y entregables pendientes

En `dev`, `app/app.py` es todavía un esqueleto y no hay un champion serializado
en `models/`. No se publica una URL de despliegue ni se presenta la app como
operativa. La integración de la app, feedback, MongoDB, Docker, reentrenamiento
y drift se entregan en sus propias issues. Sus contratos de ejecución deben
documentarse al integrarse; no son requisitos para reproducir el split.

La elección del champion corresponde a `notebooks/08_comparison.ipynb` cuando
estén disponibles los resultados comparables de los modelos. No asumir que
el primer modelo entrenado es el champion.

## Estructura y colaboración

```text
app/          Aplicación Streamlit (en desarrollo)
data/raw/     Snapshot descargado localmente; ignorado por Git
data/processed/  Datos derivados locales; ignorados por Git
models/       Artefactos de modelos, cuando se integren
notebooks/    Preparación, análisis y comparación de modelos
src/          Limpieza, selección de variables y preprocesamiento compartidos
tests/        Tests con pytest
docs/         Decisiones, pruebas manuales e informe técnico
```

Trabajar desde `dev` actualizado en una rama por issue. Revisar asignaciones y
PRs abiertas antes de editar archivos compartidos. Seguir
[AGENTS.md](AGENTS.md) y [docs/decisiones.md](docs/decisiones.md); no cambiar
contratos comunes ni trabajo ajeno para hacer pasar una entrega. Commit, push,
PR y merge requieren la autorización correspondiente. Solicitar revisión humana
antes de integrar cambios en `dev`.
