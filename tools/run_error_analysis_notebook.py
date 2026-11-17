"""Execute only comparison + issue 17 fixed-model OOF diagnostics, not Optuna."""

from pathlib import Path
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/08_comparison.ipynb"
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=900, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, path)
    print("Issue 17: comparación y diagnóstico ejecutados sin errores; sin nueva búsqueda, test ni champion.")


if __name__ == "__main__":
    main()
