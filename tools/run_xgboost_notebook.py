"""Execute only the XGBoost notebook with the invoking Python environment.

Reads local raw data and replaces the notebook outputs/aggregate XGBoost
results. No external services, serialized models or individual predictions.
"""

from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/06_model_xgboost.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    client = NotebookClient(
        notebook, km=manager, timeout=1800,
        resources={"metadata": {"path": str(root)}},
    )
    try:
        client.execute()
    finally:
        # Preserve failed-cell evidence too; never label a failed run successful.
        nbformat.write(notebook, path)
    print("XGBoost: notebook ejecutado completo, sin errores de celda.")


if __name__ == "__main__":
    main()
