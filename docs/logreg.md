# Reproducir la baseline de regresión logística

Ejecutar los comandos desde la raíz del repositorio.

Usar Python 3.12 y el snapshot Barcelona 2026-09-17 del [README](../README.md), sin añadir datos a Git:

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
Los diagnósticos por clase y las matrices se calculan con predicciones OOF
sobre train; este notebook no evalúa el test reservado al champion en #22.
Ver [decisiones de la baseline](decisiones.md#baseline-de-regresión-logística-18)
para scoring, gap, presupuesto y limitaciones.
