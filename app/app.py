"""Streamlit app: form -> price class prediction (#15).

Run from the repo root:  streamlit run app/app.py
"""

import os
import random
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app_logic import (
    BATHROOM_OPTIONS,
    ROOM_TYPES_ES,
    build_row,
    form_to_values,
    customer_notice,
    model_details,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for src.feedback_store
from src.ab_test import assign_variant, challenger_share, variant_artifact  # noqa: E402
from src.feedback_store import StorageError, save_feedback, save_prediction  # noqa: E402

# APP_MODEL_PATH lets the tests use a small synthetic model.
MODEL_PATH = Path(
    os.environ.get(
        "APP_MODEL_PATH",
        Path(__file__).resolve().parents[1] / "models" / "app_model.joblib",
    )
)
CLASSES = ("Económico", "Medio", "Premium")

# The 10 amenities of MODEL_FEATURES: simple yes/no questions a host can answer.
AMENITIES = {
    "has_air_conditioning": "Aire acondicionado",
    "has_heating": "Calefacción",
    "has_hair_dryer": "Secador de pelo",
    "has_tv": "TV",
    "has_elevator": "Ascensor",
    "has_dishwasher": "Lavavajillas",
    "has_dedicated_workspace": "Espacio de trabajo",
    "has_self_checkin": "Self check-in",
    "has_parking": "Parking",
    "has_pool": "Piscina",
}
# Checked by default: amenities present in most Barcelona listings (> 50 % in train).
DEFAULT_AMENITIES = {"has_air_conditioning", "has_heating", "has_hair_dryer", "has_tv", "has_elevator"}

PROPERTY_TYPES_ES = {
    "Entire rental unit": "Apartamento entero",
    "Private room in rental unit": "Habitación privada en un apartamento",
    "Room in hotel": "Habitación de hotel",
    "Entire serviced apartment": "Apartamento con servicios (entero)",
    "Entire condo": "Piso en edificio residencial (entero)",
    "Private room in hostel": "Habitación privada en un hostal",
    "Entire loft": "Loft entero",
    "Room in boutique hotel": "Habitación en un hotel boutique",
    "Private room in bed and breakfast": "Habitación privada en un B&B",
    "Private room in home": "Habitación privada en una casa",
    "Shared room in hostel": "Habitación compartida en un hostal",
    "Entire home": "Casa entera",
    "Private room in condo": "Habitación privada en un piso",
    "Shared room in hotel": "Habitación compartida en un hotel",
}

CLASS_STYLE = {
    # "color": soft pastel for borders, backgrounds and bars.
    # "text": a deeper tone of the same colour so the class name stays readable.
    "Económico": {"icon": "💶", "color": "#8FD3C8", "text": "#2E7D72"},
    "Medio": {"icon": "💶💶", "color": "#9CC3E8", "text": "#3A6EA5"},
    "Premium": {"icon": "💶💶💶", "color": "#C3B5E3", "text": "#6B55A3"},
}


@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)


def result_card(prediction: str, price_range: str, probability: float) -> str:
    style = CLASS_STYLE[prediction]
    return f"""
    <div style="border:2px solid {style['color']}; border-radius:16px;
                padding:24px; text-align:center;
                background:{style['color']}33; margin-bottom:16px;">
        <div style="font-size:2.2rem;">{style['icon']}</div>
        <div style="font-size:2.4rem; font-weight:800; color:{style['text']};">
            {prediction}
        </div>
        <div style="font-size:1.1rem; margin-top:4px;">{price_range}</div>
        <div style="opacity:0.7; margin-top:8px;">Probabilidad estimada por el modelo: {probability:.0%}</div>
    </div>
    """


def probability_bars(probabilities: pd.Series) -> str:
    rows = []
    for label in CLASSES:
        value = probabilities.get(label, 0.0)
        color = CLASS_STYLE[label]["color"]
        rows.append(
            f"""
            <div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between;">
                    <span>{label}</span><b>{value:.0%}</b>
                </div>
                <div style="background:rgba(128,128,128,0.2); border-radius:6px; height:12px;">
                    <div style="width:{value * 100:.1f}%; background:{color};
                                height:12px; border-radius:6px;"></div>
                </div>
            </div>
            """
        )
    return "".join(rows)


st.set_page_config(page_title="Precio Airbnb Barcelona", page_icon="🏠", layout="wide")

if not MODEL_PATH.exists():
    st.error("No se encuentra el modelo. Ejecuta primero: `python -m src.train_app_model`")
    st.stop()

artifact = load_artifact(str(MODEL_PATH))
# A/B test (#35): one coin toss per session, kept while the session lasts.
if "variant" not in st.session_state:
    try:
        share = challenger_share(artifact)
    except ValueError as error:  # bad AB_SPLIT: log it and serve only the champion
        print(f"A/B test disabled: {error}", file=sys.stderr)
        share = 0.0
    st.session_state["variant"] = assign_variant(random.random(), share)
model_artifact = variant_artifact(artifact, st.session_state["variant"])
low, high = artifact["thresholds"]["lower"], artifact["thresholds"]["upper"]
def euros(value: float) -> str:
    return f"{value:.2f}".replace(".", ",") + " €"


# Same rule as the target: Económico <= lower < Medio <= upper < Premium.
ranges = {
    "Económico": f"hasta {euros(low)}/noche",
    "Medio": f"más de {euros(low)} y hasta {euros(high)}/noche",
    "Premium": f"más de {euros(high)}/noche",
}

st.title("🏠 ¿En qué rango de precio está tu alojamiento en Barcelona?", anchor=False)
st.caption("Describe tu anuncio de Airbnb y lo clasificamos como Económico, Medio o Premium.")

form_col, result_col = st.columns([3, 2], gap="large")

with form_col:
    with st.form("listing"):
        st.subheader("📍 Ubicación y tipo", anchor=False)
        # Official (Catalan) names, as on maps and on Airbnb, with the district
        # next to them: users who do not know the neighbourhood find it by its
        # district. Typing a district name in the box also filters the list.
        district = artifact["hood_to_district"]
        hood = st.selectbox(
            "Barrio",
            sorted(artifact["neighbourhoods"], key=lambda h: (district[h], h.lower())),
            format_func=lambda h: f"{h[0].upper()}{h[1:]} ({district[h]})",
        )
        property_type = st.selectbox(
            "Tipo de alojamiento",
            artifact["property_types"],
            format_func=lambda value: PROPERTY_TYPES_ES.get(value, value),
        )
        room_type = st.selectbox(
            "Qué se alquila",
            artifact["room_types"],
            format_func=lambda value: ROOM_TYPES_ES.get(value, value),
        )

        st.subheader("🛏️ Capacidad", anchor=False)
        c1, c2 = st.columns(2)
        accommodates = c1.number_input("Huéspedes", 1, 16, 2)
        bedrooms = c2.number_input("Dormitorios", 0, 10, 1, help="0 si es un estudio.")
        beds = c1.number_input("Camas", 1, 20, 1)
        bathrooms = c2.number_input("Baños", 0.0, 10.0, 1.0, step=0.5, format="%.1f")
        bathroom = st.radio(
            "Tipo de baño",
            list(BATHROOM_OPTIONS),
            format_func=BATHROOM_OPTIONS.get,
            horizontal=True,
        )
        min_nights = st.number_input("Noches mínimas", 1, 365, 2)

        st.subheader("✨ Servicios", anchor=False)
        st.caption("Marca los que ofrece tu alojamiento.")
        cols = st.columns(2)
        flags = {
            key: cols[i % 2].checkbox(label, value=key in DEFAULT_AMENITIES)
            for i, (key, label) in enumerate(AMENITIES.items())
        }

        submitted = st.form_submit_button("🔮 Predecir", type="primary", width="stretch")

with result_col:
    st.subheader("🏷️ Resultado", anchor=False)
    if not submitted and "last" not in st.session_state:
        st.info("Rellena el formulario y pulsa **Predecir** para ver el rango de precio.")
    if submitted:
        form = {
            "hood": hood,
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "minimum_nights": min_nights,
            "bathrooms_count": bathrooms,
            "bathroom": bathroom,
            "amenities": flags,
        }
        try:
            values = form_to_values(artifact, form)
        except ValueError as error:
            st.error(str(error))
            st.stop()

        prediction, probabilities = predict(model_artifact, build_row(artifact, values))
        # Every prediction is stored (#23): #25 metrics, #36 drift, #27 MongoDB.
        try:
            prediction_id = save_prediction(
                values, prediction, probabilities.to_dict(),
                model_name=model_artifact["model_name"],
                model_version=model_artifact.get("model_version", "desconocida"),
                thresholds=artifact["thresholds"],
                variant=st.session_state["variant"],  # A/B test (#35)
            )
        except StorageError:
            prediction_id = None  # storage unavailable: still show the result
        st.session_state["last"] = {
            "prediction": prediction,
            "probabilities": probabilities.to_dict(),
            "prediction_id": prediction_id,
            "feedback": None,
        }

    last = st.session_state.get("last")
    if last:
        prediction = last["prediction"]
        probabilities = pd.Series(last["probabilities"])
        st.markdown(
            result_card(prediction, ranges[prediction], probabilities[prediction]),
            unsafe_allow_html=True,
        )
        st.caption(
            "Límites de las categorías calculados con los precios de "
            "entrenamiento y redondeados al céntimo."
        )
        st.markdown("**Probabilidad por clase**")
        st.markdown(probability_bars(probabilities), unsafe_allow_html=True)
        st.caption(customer_notice(model_artifact))
        with st.expander("ℹ️ Sobre el modelo"):
            st.markdown(model_details(model_artifact))

        st.subheader("💬 ¿Nos ayudas a mejorar?", anchor=False)
        if last["feedback"]:
            outcome = "acertó" if last["feedback"]["correct"] else "no acertó"
            st.success(
                f"¡Gracias! Con {euros(last['feedback']['real_price'])} la categoría real es "
                f"**{last['feedback']['real_class']}**: el modelo {outcome}."
            )
        elif last["prediction_id"] is None:
            st.caption("No se ha podido guardar la predicción, así que no se puede enviar respuesta.")
        else:
            with st.form("feedback"):
                real_price = st.number_input(
                    "Precio real por noche de tu anuncio (€)",
                    min_value=0.0, value=None, step=1.0, format="%.2f",
                    help="Opcional. Solo se guarda el precio y las respuestas del formulario, "
                         "sin datos personales.",
                )
                sent = st.form_submit_button("Enviar")
            if sent:
                try:
                    last["feedback"] = save_feedback(last["prediction_id"], real_price)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
                except StorageError:
                    # Keep the result on screen and do not confirm a failed save.
                    st.warning("No hemos podido guardar tu respuesta. Inténtalo de nuevo más tarde.")

st.divider()
st.caption(
    "Datos: [Inside Airbnb](https://insideairbnb.com/get-the-data/) · "
    "Barcelona, septiembre de 2026 · CC BY 4.0"
)
