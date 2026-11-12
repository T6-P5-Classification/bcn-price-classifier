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
| Variables de entrada | Train y test tienen las mismas columnas, orden y tipos; no entran precio, identificadores ni los campos excluidos comprobados. Se conservan las variables derivadas esperadas. |
| Nulos permitidos | Los valores vacíos de `minimum_nights` se conservan con su indicador; no se rellenan antes del preprocesamiento. |
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
Tampoco fijan como requisito el número de filas o las 71 variables del
snapshot actual.

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
