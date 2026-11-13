"""Streamlit app: form -> price class prediction (#15).

Run from the repo root:  streamlit run app/app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "app_model.joblib"
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
def load_artifact():
    return joblib.load(MODEL_PATH)


def build_row(artifact: dict, values: dict) -> pd.DataFrame:
    """One-row DataFrame with every model column.

    The form must fill all 23 model variables: a missing one would be imputed
    silently by the pipeline (worse predictions), so it raises an error instead.
    """
    missing = set(artifact["columns"]) - set(values)
    if missing:
        raise ValueError(f"Form is missing model variables: {sorted(missing)}")
    row = {column: values[column] for column in artifact["columns"]}
    df = pd.DataFrame([row], columns=artifact["columns"])
    numeric = artifact["numeric_columns"]
    df[numeric] = df[numeric].astype(float)
    other = [c for c in df.columns if c not in numeric]
    df[other] = df[other].astype(object)
    return df


def result_card(prediction: str, price_range: str, confidence: float) -> str:
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
        <div style="opacity:0.7; margin-top:8px;">Confianza del modelo: {confidence:.0%}</div>
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

artifact = load_artifact()
low, high = artifact["thresholds"]["lower"], artifact["thresholds"]["upper"]
ranges = {
    "Económico": f"hasta {low:.0f} €/noche",
    "Medio": f"entre {low:.0f} y {high:.0f} €/noche",
    "Premium": f"más de {high:.0f} €/noche",
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

        st.subheader("🛏️ Capacidad", anchor=False)
        c1, c2 = st.columns(2)
        accommodates = c1.number_input("Huéspedes", 1, 16, 2)
        bedrooms = c2.number_input("Dormitorios", 0, 10, 1, help="0 si es un estudio.")
        beds = c1.number_input("Camas", 1, 20, 1)
        bathrooms = c2.number_input("Baños", 0.0, 10.0, 1.0, step=0.5, format="%.1f")
        shared_bath = st.checkbox(
            "Baño compartido con otras personas",
            help="Solo para habitaciones. En un alojamiento entero no se aplica.",
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
    if not submitted:
        st.info("Rellena el formulario y pulsa **Predecir** para ver el rango de precio.")
    else:
        coords = artifact["hood_coords"][hood]
        room_type = artifact["property_to_room"][property_type]
        is_room = room_type != "Entire home/apt"
        # Same pattern as the training data: a shared room always has a shared
        # bathroom, and "private bathroom" only applies to rooms.
        is_shared = room_type == "Shared room" or (is_room and shared_bath)
        values = {
            "neighbourhood_cleansed": hood,
            "neighbourhood_group_cleansed": artifact["hood_to_district"][hood],
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "property_type": property_type,
            "room_type": room_type,
            "accommodates": accommodates,
            "bedrooms": bedrooms,
            "beds": beds,
            "minimum_nights": min_nights,
            "bathrooms_count": bathrooms,
            "is_shared_bathroom": int(is_shared),
            "is_private_bathroom": int(is_room and not is_shared),
            **{key: int(value) for key, value in flags.items()},
        }

        row = build_row(artifact, values)
        model = artifact["model"]
        prediction = model.predict(row)[0]
        probabilities = pd.Series(model.predict_proba(row)[0], index=model.classes_)

        st.markdown(
            result_card(prediction, ranges[prediction], probabilities[prediction]),
            unsafe_allow_html=True,
        )
        st.markdown("**Probabilidad por clase**")
        st.markdown(probability_bars(probabilities), unsafe_allow_html=True)

st.divider()
st.caption(
    "Datos: [Inside Airbnb](https://insideairbnb.com/get-the-data/) · "
    "Barcelona, septiembre de 2026 · CC BY 4.0"
)
