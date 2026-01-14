from typing import Dict, Tuple

from flask import Flask, jsonify, request
from flask_cors import CORS

from model_utils import MODEL_DESCRIPTIONS, MODEL_LABELS, summarize_prediction, get_or_train_models

app = Flask(__name__)
CORS(app)

# Load only Logistic Regression model
print("\n" + "=" * 70)
print("CARDIOVASCULAR RISK ASSESSMENT API - SERVER INITIALIZATION")
print("=" * 70)
print("\n🔄 Loading model...")

models = get_or_train_models()
LOGISTIC_MODEL_KEY = 'logistic_regression'

# Display loaded model information
if LOGISTIC_MODEL_KEY in models:
    model = models[LOGISTIC_MODEL_KEY]
    print("\n✅ MODEL LOADED SUCCESSFULLY!")
    print("-" * 70)
    print(f"Model Name:        {MODEL_LABELS[LOGISTIC_MODEL_KEY]}")
    print(f"Model Type:        {'Regression' if model.is_regression else 'Classification'}")
    print(f"Description:       {MODEL_DESCRIPTIONS[LOGISTIC_MODEL_KEY]}")
    print("\n📊 Model Performance Metrics:")
    print(f"   • Accuracy:     {model.metrics['accuracy']:.2%}")
    print(f"   • Precision:    {model.metrics['precision']:.2%}")
    print(f"   • Recall:       {model.metrics['recall']:.2%}")
    print(f"   • F1 Score:     {model.metrics['f1']:.2%}")
    print("-" * 70)
    print("\n✅ Server is ready to accept requests!")
    print(f"   • Health Check:  GET  http://localhost:5000/health")
    print(f"   • Prediction:    POST http://localhost:5000/predict")
    print("\n" + "=" * 70 + "\n")
else:
    print("\n❌ ERROR: Logistic Regression model not found!")
    print("=" * 70 + "\n")


REQUIRED_FIELDS = [
    "age",
    "gender",
    "blood_pressure",
    "cholesterol",
    "heart_rate",
    "diabetes",
    "smoking",
    "obesity",
]


ALLOWED_GENDERS = {"male", "female"}


def _is_number(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_payload(payload: Dict[str, object]) -> Tuple[bool, str]:
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        return False, f"Missing fields: {', '.join(missing)}"

    gender = str(payload.get("gender", "")).lower()
    if gender not in ALLOWED_GENDERS:
        return False, "Gender must be 'male' or 'female'."

    numeric_fields = ["age", "blood_pressure", "cholesterol", "heart_rate"]
    for field in numeric_fields:
        if not _is_number(payload.get(field)):
            return False, f"Field '{field}' must be a number."
    return True, ""


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "model": MODEL_LABELS[LOGISTIC_MODEL_KEY]}


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400

    result = summarize_prediction(LOGISTIC_MODEL_KEY, models[LOGISTIC_MODEL_KEY], payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
