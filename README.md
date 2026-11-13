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
## Reproducir la baseline de regresión logística

Usar Python 3.12 y el snapshot indicado arriba, sin añadir datos a Git:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-notebooks.txt
python tools/run_notebook.py
python -m pytest -q
```

El script ejecuta de arriba abajo `04_model_logreg.ipynb` y después
`08_comparison.ipynb` usando el mismo intérprete que lo lanza. El kernel y
las cachés se guardan en `.jupyter/` (ignorado). En VS Code, abrir el notebook,
usar **Select Kernel → Python Environments** y elegir el Python de `.venv`.
La ejecución vuelve a generar las métricas agregadas en `docs/results/`.
Ver [decisiones de la baseline](docs/decisiones.md#baseline-de-regresión-logística-18)
para scoring, gap, presupuesto y limitaciones.
