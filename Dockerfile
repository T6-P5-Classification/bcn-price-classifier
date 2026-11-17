# App Streamlit con el champion XGBoost (#29).
#
# El modelo no está en Git: genéralo antes de construir la imagen
#   python -m src.train_app_model
#   docker build -t bcn-price-classifier .
#   docker run -p 8501:8501 bcn-price-classifier
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# XGBoost necesita la librería OpenMP del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero: se reutilizan en caché si solo cambia el código
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

COPY .streamlit/ .streamlit/
COPY src/ src/
COPY app/ app/
COPY models/app_model.joblib models/app_model.joblib

# Usuario sin privilegios; carpeta escribible para datos que genere la app
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p data/feedback \
    && chown -R appuser:appuser data
USER appuser

# Render y otros servicios indican el puerto con PORT; en local, 8501
ENV PORT=8501
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/_stcore/health')"

CMD ["sh", "-c", "streamlit run app/app.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true --browser.gatherUsageStats=false"]
