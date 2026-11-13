"""Execute the baseline and comparison notebooks with this Python environment."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.jupyter'
os.environ['JUPYTER_PATH'] = str(LOCAL / 'share' / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(LOCAL / 'runtime')
os.environ['MPLCONFIGDIR'] = str(LOCAL / 'matplotlib')
os.environ['IPYTHONDIR'] = str(LOCAL / 'ipython')

import nbformat
from nbclient import NotebookClient
from ipykernel.kernelspec import install

install(prefix=str(LOCAL), kernel_name='python3', display_name='Python (.venv)')
for name in sys.argv[1:] or ['04_model_logreg', '08_comparison']:
    path = ROOT / 'notebooks' / (name + '.ipynb')
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(
            notebook, timeout=1800, kernel_name='python3',
            resources={'metadata': {'path': str(ROOT / 'notebooks')}},
        ).execute()
    finally:
        nbformat.write(notebook, path)
    print(f'{name}: executed successfully', flush=True)
