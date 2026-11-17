"""Execute issue 11 locally; overwrite only its notebook and aggregate EDA outputs."""

from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/03_eda_map.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=300, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("EDA #11: notebook completo, sin errores de celda; sin entrenar modelos ni evaluar test.")


if __name__ == "__main__":
    main()
