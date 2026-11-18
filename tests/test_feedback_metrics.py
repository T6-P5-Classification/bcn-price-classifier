"""Metrics from the shared feedback contract, checked by hand."""
import pandas as pd
import pytest

from src.feedback_metrics import feedback_metrics


def frames():
    predictions = pd.DataFrame([
        dict(prediction_id=i, model_name="XGBoost", model_version="v1", variant="champion",
             threshold_lower=122., threshold_upper=250., predicted_class=label)
        for i, label in zip("abcde", ["Económico", "Premium", "Premium", "Económico", "Medio"])
    ])
    labelled = predictions.iloc[:4].copy()
    labelled["real_class"] = ["Económico", "Medio", "Premium", "Premium"]
    labelled["real_price"] = [100., 200., 300., 300.]
    return predictions, labelled


def test_scores_denominator_and_real_predicted_orientation():
    predictions, labelled = frames()
    result = next(iter(feedback_metrics(predictions, labelled).values()))
    assert result["n_predictions"] == 5 and result["n_feedback"] == 4
    assert result["coverage"] == pytest.approx(0.8)
    assert result["accuracy"] == 0.5
    assert result["f1_macro"] == pytest.approx(7 / 18)
    assert result["extreme_error_rate"] == 0.25
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
    assert result["per_class"]["Medio"]["support"] == 1
    assert result["prediction_counts"] == {"Económico": 2, "Medio": 1, "Premium": 2}


@pytest.mark.parametrize("field,value", [("model_version", "v2"), ("variant", "challenger"), ("threshold_upper", 260.)])
def test_model_variants_and_thresholds_are_separate_cohorts(field, value):
    predictions, labelled = frames()
    predictions.loc[predictions.prediction_id == "e", field] = value
    results = feedback_metrics(predictions, labelled)
    assert len(results) == 2
    assert sorted(x["n_predictions"] for x in results.values()) == [1, 4]
    assert sorted(x["n_feedback"] for x in results.values()) == [0, 4]


def test_no_feedback_has_no_accuracy_or_f1():
    predictions, labelled = frames()
    result = next(iter(feedback_metrics(predictions, labelled.iloc[:0]).values()))
    assert result["coverage"] == 0
    assert result["accuracy"] is None and result["f1_macro"] is None
    assert result["n_feedback"] == 0


def test_empty_store_has_no_cohorts():
    predictions, labelled = frames()
    assert feedback_metrics(predictions.iloc[:0], labelled.iloc[:0]) == {}


def test_all_three_classes_keep_equal_weight_when_support_is_missing():
    predictions, labelled = frames()
    result = next(iter(feedback_metrics(predictions.iloc[:1], labelled.iloc[:1]).values()))
    assert result["accuracy"] == 1
    assert result["f1_macro"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("mutation", ["duplicate_prediction", "duplicate_feedback", "unknown_id", "wrong_prediction", "invalid_label", "wrong_price_class", "bad_threshold"])
def test_inconsistent_data_stops_metrics_instead_of_inflating_them(mutation):
    predictions, labelled = frames()
    if mutation == "duplicate_prediction": predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif mutation == "duplicate_feedback": labelled = pd.concat([labelled, labelled.iloc[:1]])
    elif mutation == "unknown_id": labelled.loc[0, "prediction_id"] = "unknown"
    elif mutation == "wrong_prediction": labelled.loc[0, "predicted_class"] = "Medio"
    elif mutation == "invalid_label": predictions.loc[0, "predicted_class"] = "Luxury"
    elif mutation == "wrong_price_class": labelled.loc[0, "real_class"] = "Premium"
    else: predictions.loc[0, "threshold_upper"] = 100.
    with pytest.raises(ValueError): feedback_metrics(predictions, labelled)


def test_does_not_mutate_or_expose_individual_records():
    predictions, labelled = frames()
    predictions["private_extra"] = "secret-example"
    original = predictions.copy(deep=True)
    result = feedback_metrics(predictions, labelled)
    pd.testing.assert_frame_equal(original, predictions)
    assert "secret-example" not in repr(result)
    assert "prediction_id" not in repr(result)
