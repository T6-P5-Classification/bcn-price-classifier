"""Live aggregate feedback view. Storage stays behind the shared readers."""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.feedback_metrics import feedback_metrics
from src.feedback_store import StorageError, load_labelled, load_predictions
from src.preprocessing import TARGET_LABELS


def percent(value):
    return f"{value:.1%}".replace(".", ",").replace("%", " %")


@st.fragment(run_every="10s")
def render_feedback_panel():
    """Reload while the page is active; no background job and no write calls."""
    st.button("Actualizar ahora", key="feedback_refresh")
    try:
        # Read answers first. With append-only storage, the later predictions
        # snapshot includes every prediction already present in those answers.
        labelled = load_labelled()
        predictions = load_predictions()
        summaries = feedback_metrics(predictions, labelled)
    except StorageError:
        st.warning("No se puede leer el feedback ahora. Inténtalo de nuevo más tarde.")
        return
    except ValueError:
        st.warning("Los registros de feedback no son consistentes. No se muestran métricas.")
        return
    st.caption("Actualizado: " + datetime.now(timezone.utc).strftime("%d/%m/%Y %H:%M:%S UTC"))
    st.caption(
        "Respuestas voluntarias y precios declarados sin verificar. Estas métricas "
        "describen a quienes responden y no sustituyen la evaluación del modelo."
    )
    if not summaries:
        st.info("No hay predicciones guardadas todavía.")
        return
    cohort = st.selectbox(
        "Modelo, versión y categorías de precio", list(summaries),
        format_func=lambda x: f"{x[0]} · {x[1]} · {x[2]} · límites {x[3]:g} / {x[4]:g} €",
        key="feedback_cohort",
    )
    result = summaries[cohort]
    counts = st.columns(3)
    counts[0].metric("Predicciones guardadas", str(result["n_predictions"]))
    counts[1].metric("Respuestas", str(result["n_feedback"]))
    counts[2].metric("Cobertura del feedback", percent(result["coverage"]))
    st.caption("Cobertura: respuestas divididas entre predicciones guardadas de esta versión y variante.")
    st.dataframe(pd.DataFrame({"Predicciones": result["prediction_counts"]}).rename_axis("Clase predicha"))
    if not result["n_feedback"]:
        st.info("Esta versión sigue sin respuestas con precio real; aún no hay métricas de acierto.")
        return
    scores = st.columns(3)
    scores[0].metric("Accuracy", percent(result["accuracy"]))
    scores[1].metric("F1 macro", f"{result['f1_macro']:.3f}".replace(".", ","))
    scores[2].metric("Errores extremos", percent(result["extreme_error_rate"]))
    st.caption(
        f"Calculado sobre {result['n_feedback']} respuestas. Macro F1 da el mismo peso "
        "a las tres clases; las clases sin ejemplos o predicciones tienen F1 igual a 0. "
        "Los errores extremos confunden Económico con Premium o al revés."
    )
    st.markdown("**Métricas por clase real según el precio declarado**")
    report = pd.DataFrame(result["per_class"]).T[["precision", "recall", "f1-score", "support"]]
    report["support"] = report["support"].astype(int)
    st.dataframe(report.rename_axis("Clase real"))
    st.markdown("**Matriz de confusión del feedback**")
    st.dataframe(pd.DataFrame(
        result["confusion_matrix"],
        index=pd.Index(TARGET_LABELS, name="Clase real según precio declarado"),
        columns=pd.Index(TARGET_LABELS, name="Clase predicha"),
    ))
