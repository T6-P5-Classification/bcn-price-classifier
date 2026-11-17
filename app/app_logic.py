"""Form logic of the Streamlit app, kept apart from the UI so it can be tested.

Nothing here guesses a value the user can declare: room type and bathroom are
asked in the form. Combinations that never occur in the training data are
rejected with a clear message instead of being silently corrected.
"""

from __future__ import annotations

import pandas as pd

ENTIRE = "Entire home/apt"
SHARED_ROOM = "Shared room"

ROOM_TYPES_ES = {
    "Entire home/apt": "Alojamiento entero",
    "Private room": "Habitación privada",
    "Hotel room": "Habitación de hotel",
    "Shared room": "Habitación compartida",
}

# Bathroom options in the form -> (is_shared_bathroom, is_private_bathroom),
# the same flags that src/cleaning.py derives from Airbnb's bathrooms_text:
# "1 bath" -> (0, 0), "1 private bath" -> (0, 1), "1 shared bath" -> (1, 0).
BATHROOM_OPTIONS = {
    "own": "Propio del alojamiento entero",
    "private": "Privado, solo para la habitación",
    "shared": "Compartido con otras personas",
}
BATHROOM_FLAGS = {"own": (0, 0), "private": (0, 1), "shared": (1, 0)}


def bathroom_flags(room_type: str, bathroom: str) -> tuple[int, int]:
    """Return (is_shared_bathroom, is_private_bathroom) or explain the conflict.

    In train every entire home has (0, 0) and every shared room (1, 0). For
    private and hotel rooms, (0, 0) also exists (216 and 6 listings): it means
    Airbnb's text said neither "private" nor "shared", not that the bathroom
    belongs to an entire home. The form asks the user to clarify instead of
    offering an "unspecified" answer, so for rooms "own" is not accepted.
    """
    if bathroom not in BATHROOM_FLAGS:
        raise ValueError(f"Unknown bathroom option: {bathroom!r}")
    if room_type == ENTIRE and bathroom != "own":
        raise ValueError(
            "Un alojamiento entero tiene baño propio: elige «Propio del alojamiento entero»."
        )
    if room_type != ENTIRE and bathroom == "own":
        raise ValueError(
            "Para una habitación, indica si el baño es privado o compartido."
        )
    if room_type == SHARED_ROOM and bathroom != "shared":
        raise ValueError(
            "En una habitación compartida el baño también es compartido: "
            "elige «Compartido con otras personas»."
        )
    return BATHROOM_FLAGS[bathroom]


def check_room_type(artifact: dict, property_type: str, room_type: str) -> None:
    """Reject a room type never seen with this property type in train.

    Unambiguous types allow one option ("Entire rental unit" is always rented
    whole); ambiguous ones keep all their observed options ("Room in hotel":
    entire, hotel room or private room), none imposed by frequency.
    """
    allowed = artifact["room_types_by_property"][property_type]
    if room_type not in allowed:
        options = " o ".join(f"«{ROOM_TYPES_ES.get(r, r)}»" for r in allowed)
        raise ValueError(
            f"«{ROOM_TYPES_ES.get(room_type, room_type)}» no encaja con este tipo de "
            f"alojamiento: en los datos de Barcelona se alquila como {options}."
        )


def form_to_values(artifact: dict, form: dict) -> dict:
    """Map the answers of the form to the 23 model variables."""
    check_room_type(artifact, form["property_type"], form["room_type"])
    coords = artifact["hood_coords"][form["hood"]]
    is_shared, is_private = bathroom_flags(form["room_type"], form["bathroom"])
    return {
        "neighbourhood_cleansed": form["hood"],
        "neighbourhood_group_cleansed": artifact["hood_to_district"][form["hood"]],
        "latitude": coords["latitude"],
        "longitude": coords["longitude"],
        "property_type": form["property_type"],
        "room_type": form["room_type"],
        "accommodates": form["accommodates"],
        "bedrooms": form["bedrooms"],
        "beds": form["beds"],
        "minimum_nights": form["minimum_nights"],
        "bathrooms_count": form["bathrooms_count"],
        "is_shared_bathroom": is_shared,
        "is_private_bathroom": is_private,
        **{key: int(bool(value)) for key, value in form["amenities"].items()},
    }


def build_row(artifact: dict, values: dict) -> pd.DataFrame:
    """One-row DataFrame with every model column.

    The form must fill all 23 model variables: a missing or empty one would be
    imputed silently by the pipeline, so it raises an error instead.
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
    # Checked after the conversion, so any kind of missing value is caught
    # (None, float NaN, numpy NaN of any precision, pandas NA).
    empty = df.columns[df.isna().iloc[0]].tolist()
    if empty:
        raise ValueError(f"Form left model variables empty: {empty}")
    return df


def predict(artifact: dict, row: pd.DataFrame) -> tuple[str, pd.Series]:
    """Predicted class name and probabilities indexed by class name.

    Models can output text labels (logistic regression) or numbers (XGBoost
    uses 0/1/2): ``class_labels`` in the artifact translates both.
    """
    model = artifact["model"]
    labels = artifact["class_labels"]
    probabilities = pd.Series(
        model.predict_proba(row)[0], index=[labels[c] for c in model.classes_]
    )
    return labels[model.predict(row)[0]], probabilities
