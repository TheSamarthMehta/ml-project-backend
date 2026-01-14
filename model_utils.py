import math
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


@dataclass
class ModelBundle:
    pipeline: Pipeline
    is_regression: bool
    metrics: Dict[str, float]


MODEL_LABELS: Dict[str, str] = {
    "linear_regression": "Linear Regression",
    "polynomial_regression": "Polynomial Regression",
    "logistic_regression": "Logistic Regression",
    "knn": "K-Nearest Neighbors",
    "svm": "Support Vector Machine",
    "ann": "Artificial Neural Network",
}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "linear_regression": "Learns straight-line risk trends from your numbers.",
    "polynomial_regression": "Captures gentle curves in risk patterns for nuance.",
    "logistic_regression": "Classic medical-style classifier for heart risk.",
    "knn": "Finds people like you and mirrors their outcomes.",
    "svm": "Separates low vs high risk with a flexible margin.",
    "ann": "Learns layered patterns similar to how neurons connect.",
}

HEALTHY_RANGES: Dict[str, Tuple[float, float]] = {
    "blood_pressure": (95.0, 129.0),
    "cholesterol": (150.0, 210.0),
    "heart_rate": (60.0, 95.0),
    "age": (35.0, 65.0),
}

RISK_LEVELS = [
    (0.0, 0.33, "Low"),
    (0.33, 0.66, "Medium"),
    (0.66, 1.01, "High"),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_dataset(n_samples: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ages = rng.integers(30, 81, size=n_samples)
    genders = rng.choice([0, 1], size=n_samples)  # 0 female, 1 male

    base_bp = rng.normal(122, 15, size=n_samples)
    cholesterol = rng.normal(205, 38, size=n_samples)
    heart_rate = rng.normal(78, 12, size=n_samples)

    diabetes = rng.binomial(1, 0.22, size=n_samples)
    smoking = rng.binomial(1, 0.28, size=n_samples)
    obesity = rng.binomial(1, 0.25, size=n_samples)

    # Risk logit leans on age, bp, cholesterol, lifestyle toggles.
    logit = (
        -6.5
        + 0.045 * (ages - 50)
        + 0.035 * (base_bp - 120)
        + 0.025 * (cholesterol - 200)
        + 0.02 * (heart_rate - 75)
        + 0.9 * diabetes
        + 0.75 * smoking
        + 0.65 * obesity
        + 0.35 * genders
    )

    probability = sigmoid(logit)
    target = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "age": ages,
            "gender": genders,
            "blood_pressure": base_bp,
            "cholesterol": cholesterol,
            "heart_rate": heart_rate,
            "diabetes": diabetes,
            "smoking": smoking,
            "obesity": obesity,
            "target": target,
        }
    )

    return df


def build_datasets(seed: int = 7):
    df = generate_synthetic_dataset(seed=seed)
    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_features = [
        "age",
        "blood_pressure",
        "cholesterol",
        "heart_rate",
    ]
    binary_features = ["gender", "diabetes", "smoking", "obesity"]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    binary_transformer = "passthrough"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
    }


def _evaluate_predictions(y_true, probas, preds) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }


def _make_metric_bundle(
    model: Pipeline, y_true, X_test, is_regression: bool
) -> Tuple[Dict[str, float], np.ndarray]:
    if is_regression:
        raw = model.predict(X_test)
        probas = sigmoid(raw)
    else:
        probas = model.predict_proba(X_test)[:, 1]
    preds = (probas >= 0.5).astype(int)
    metrics = _evaluate_predictions(y_true, probas, preds)
    return metrics, probas


def train_all_models(seed: int = 7) -> Dict[str, ModelBundle]:
    data = build_datasets(seed=seed)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    models: Dict[str, ModelBundle] = {}

    linear_reg = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linear_reg.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(linear_reg, y_test, X_test, is_regression=True)
    models["linear_regression"] = ModelBundle(linear_reg, True, metrics)

    poly_reg = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    poly_reg.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(poly_reg, y_test, X_test, is_regression=True)
    models["polynomial_regression"] = ModelBundle(poly_reg, True, metrics)

    logistic = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(logistic, y_test, X_test, is_regression=False)
    models["logistic_regression"] = ModelBundle(logistic, False, metrics)

    knn = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", KNeighborsClassifier(n_neighbors=9, weights="distance")),
        ]
    )
    knn.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(knn, y_test, X_test, is_regression=False)
    models["knn"] = ModelBundle(knn, False, metrics)

    svm = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                SVC(
                    probability=True,
                    kernel="rbf",
                    class_weight="balanced",
                    gamma="scale",
                    random_state=seed,
                ),
            ),
        ]
    )
    svm.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(svm, y_test, X_test, is_regression=False)
    models["svm"] = ModelBundle(svm, False, metrics)

    ann = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(18, 10),
                    max_iter=500,
                    random_state=seed,
                    learning_rate_init=0.01,
                ),
            ),
        ]
    )
    ann.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(ann, y_test, X_test, is_regression=False)
    models["ann"] = ModelBundle(ann, False, metrics)

    return models


def save_models_to_disk(models: Dict[str, ModelBundle], models_dir: str = "models"):
    """Save all trained models to disk as .pkl files."""
    os.makedirs(models_dir, exist_ok=True)
    
    for model_name, model_bundle in models.items():
        filename = f"{model_name.replace('_', '-')}.pkl"
        filepath = os.path.join(models_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_bundle, f)
        
        print(f"✓ Saved {model_name} to {filepath}")
    
    print(f"\n✓ All {len(models)} models saved successfully!")


def load_models_from_disk(models_dir: str = "models", only_logistic: bool = False) -> Dict[str, ModelBundle]:
    """Load trained models from disk."""
    models = {}
    
    if not os.path.exists(models_dir):
        return None
    
    if only_logistic:
        # Load only logistic regression model
        model_files = {
            "logistic-regression.pkl": "logistic_regression",
        }
    else:
        # Load all models
        model_files = {
            "logistic-regression.pkl": "logistic_regression",
            "linear-regression.pkl": "linear_regression",
            "polynomial-regression.pkl": "polynomial_regression",
            "knn.pkl": "knn",
            "svm.pkl": "svm",
            "ann.pkl": "ann",
        }
    
    loaded_count = 0
    for filename, model_key in model_files.items():
        filepath = os.path.join(models_dir, filename)
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    models[model_key] = pickle.load(f)
                loaded_count += 1
            except Exception as e:
                print(f"   ❌ Failed to load {model_key}: {e}")
        else:
            if only_logistic:
                return None
    
    if loaded_count == 0:
        return None
    
    return models


def train_logistic_model(seed: int = 7) -> Dict[str, ModelBundle]:
    """Train only the logistic regression model."""
    data = build_datasets(seed=seed)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    preprocessor = data["preprocessor"]

    models: Dict[str, ModelBundle] = {}

    logistic = Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic.fit(X_train, y_train)
    metrics, _ = _make_metric_bundle(logistic, y_test, X_test, is_regression=False)
    models["logistic_regression"] = ModelBundle(logistic, False, metrics)

    return models


def get_or_train_models(models_dir: str = "models", seed: int = 7, only_logistic: bool = True) -> Dict[str, ModelBundle]:
    """Load models from disk if available, otherwise train and save them."""
    models = load_models_from_disk(models_dir, only_logistic=only_logistic)
    
    if models is None:
        print("   ⚙️  Training model from scratch...\n")
        if only_logistic:
            models = train_logistic_model(seed=seed)
        else:
            models = train_all_models(seed=seed)
        
        print("\n   💾 Saving model to disk...")
        save_models_to_disk(models, models_dir)
        print("")
    
    return models


def format_probability(prob: float) -> float:
    return float(np.clip(prob, 0.0, 1.0))


def risk_level_from_prob(prob: float) -> str:
    safe_prob = format_probability(prob)
    for low, high, label in RISK_LEVELS:
        if low <= safe_prob < high:
            return label
    return "High"


def vectorize_payload(payload: Dict[str, object]) -> pd.DataFrame:
    row = {
        "age": float(payload.get("age", 0)),
        "gender": 1.0 if str(payload.get("gender", "")).lower() == "male" else 0.0,
        "blood_pressure": float(payload.get("blood_pressure", 0)),
        "cholesterol": float(payload.get("cholesterol", 0)),
        "heart_rate": float(payload.get("heart_rate", 0)),
        "diabetes": 1.0 if payload.get("diabetes") else 0.0,
        "smoking": 1.0 if payload.get("smoking") else 0.0,
        "obesity": 1.0 if payload.get("obesity") else 0.0,
    }
    return pd.DataFrame([row])


def explain_top_factors(payload: Dict[str, object]) -> List[str]:
    factors: List[Tuple[str, float, str]] = []

    def add_factor(label: str, score: float, reason: str):
        factors.append((label, score, reason))

    age = float(payload.get("age", 0))
    bp = float(payload.get("blood_pressure", 0))
    chol = float(payload.get("cholesterol", 0))
    hr = float(payload.get("heart_rate", 0))

    age_mid = HEALTHY_RANGES["age"][1]
    add_factor(
        "Age",
        max(0.0, (age - age_mid) / 40.0),
        "Age trends upward after midlife and can increase heart strain.",
    )

    bp_high = HEALTHY_RANGES["blood_pressure"][1]
    add_factor(
        "Blood Pressure",
        max(0.0, (bp - bp_high) / bp_high),
        "Higher pressure can overwork the heart and vessels.",
    )

    chol_high = HEALTHY_RANGES["cholesterol"][1]
    add_factor(
        "Cholesterol",
        max(0.0, (chol - chol_high) / chol_high),
        "Elevated cholesterol can lead to plaque buildup.",
    )

    hr_high = HEALTHY_RANGES["heart_rate"][1]
    add_factor(
        "Heart Rate",
        max(0.0, (hr - hr_high) / hr_high),
        "A faster resting pulse can signal added cardiac workload.",
    )

    if payload.get("diabetes"):
        add_factor("Diabetes", 0.35, "Blood sugar challenges increase cardiac risk.")
    if payload.get("smoking"):
        add_factor("Smoking", 0.45, "Smoking constricts vessels and reduces oxygen.")
    if payload.get("obesity"):
        add_factor(
            "Obesity", 0.4, "Weight can raise blood pressure and cholesterol over time."
        )

    sorted_factors = sorted(factors, key=lambda x: x[1], reverse=True)
    top_three = [f[0] for f in sorted_factors[:3]]
    return top_three


def craft_message(risk_level: str, top_factors: List[str]) -> str:
    if not top_factors:
        return "Your numbers are within a steady range today."
    factors_text = " and ".join(top_factors[:2]) if len(top_factors) >= 2 else top_factors[0]
    if risk_level == "High":
        return (
            f"Higher risk driven by {factors_text}. Please consider a check-in with a clinician."
        )
    if risk_level == "Medium":
        return f"Moderate risk influenced by {factors_text}. Small changes can lower it."
    return f"Low risk. Keep up the good habits around {factors_text}."


def health_recommendations(risk_level: str) -> Dict[str, List[str]]:
    if risk_level == "High":
        return {
            "diet": [
                "Prioritize vegetables, legumes, and lean proteins each meal.",
                "Cut back on salty, fried, and processed foods this week.",
                "Swap sugary drinks for water or unsweetened tea.",
            ],
            "exercise": [
                "Aim for 30 minutes of brisk walking or cycling most days.",
                "Add two short strength sessions to support metabolism.",
            ],
            "lifestyle": [
                "Book a primary care check for blood pressure and cholesterol review.",
                "Reduce tobacco exposure and seek support if quitting.",
            ],
        }
    if risk_level == "Medium":
        return {
            "diet": [
                "Fill half your plate with produce and fiber-rich sides.",
                "Limit red meat to once weekly; choose fish or poultry instead.",
            ],
            "exercise": [
                "Target 25 minutes of moderate cardio four times weekly.",
                "Stretch or take movement breaks every hour when sitting long periods.",
            ],
            "lifestyle": [
                "Track blood pressure monthly to spot changes early.",
                "Keep caffeine moderate and prioritize 7-8 hours of sleep.",
            ],
        }
    return {
        "diet": [
            "Maintain balanced meals with whole grains and healthy fats.",
            "Stay hydrated and keep sodium modest.",
        ],
        "exercise": [
            "Keep a mix of cardio and light strength work weekly.",
            "Take daily walks after meals to support heart health.",
        ],
        "lifestyle": [
            "Continue not smoking and manage stress with short breaks.",
            "Schedule annual checkups to stay ahead of changes.",
        ],
    }


def predict_single(model: ModelBundle, payload: Dict[str, object]) -> float:
    X = vectorize_payload(payload)
    if model.is_regression:
        raw = model.pipeline.predict(X)
        prob = float(sigmoid(raw)[0])
    else:
        prob = float(model.pipeline.predict_proba(X)[0, 1])
    return format_probability(prob)


def summarize_prediction(model_name: str, model: ModelBundle, payload: Dict[str, object]) -> Dict[str, object]:
    prob = predict_single(model, payload)
    risk_level = risk_level_from_prob(prob)
    top_factors = explain_top_factors(payload)
    message = craft_message(risk_level, top_factors)
    return {
        "model": MODEL_LABELS.get(model_name, model_name),
        "model_key": model_name,
        "risk_level": risk_level,
        "probability": prob,
        "top_factors": top_factors,
        "message": message,
        "metrics": model.metrics,
        "description": MODEL_DESCRIPTIONS.get(model_name, ""),
    }
