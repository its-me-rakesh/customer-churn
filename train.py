import argparse
import joblib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_utils import load_data, prepare_xy, train_val_split
from azureml.core import Run

# ----------------------------
# Argument parser
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV")
    parser.add_argument("--model_dir", type=str, default="outputs", help="Directory to save model and metadata")
    return parser.parse_args()

# ----------------------------
# Pipeline builder
# ----------------------------
def build_pipeline():
    numeric_features = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay','Total Spend','Last Interaction']
    categorical_features = ['Gender','Subscription Type','Contract Length']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    return clf

# ----------------------------
# Model evaluation
# ----------------------------
def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': float(accuracy_score(y_val, preds)),
        'precision': float(precision_score(y_val, preds, zero_division=0)),
        'recall': float(recall_score(y_val, preds, zero_division=0)),
        'f1': float(f1_score(y_val, preds, zero_division=0)),
    }
    if probs is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_val, probs))
        except Exception:
            metrics['roc_auc'] = None
    return metrics

# ----------------------------
# Main training workflow
# ----------------------------
def main():
    args = parse_args()

    # Ensure model directory exists
    MODEL_DIR = Path(args.model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    df = load_data(args.data_path)
    X, y = prepare_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y, test_size=0.2)

    # Build pipeline and hyperparameter grid
    pipeline = build_pipeline()
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    metrics = evaluate_model(best, X_val, y_val)

    print("Best params:", grid.best_params_)
    print("Validation metrics:", metrics)

    # Save model
    model_path = MODEL_DIR / "churn_rf.joblib"
    joblib.dump(best, model_path)

    # Save metadata
    meta = {
        'model_path': str(model_path),
        'best_params': grid.best_params_,
        'metrics': metrics
    }
    with open(MODEL_DIR / "model_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Azure ML logging and model registration
    run = Run.get_context()
    for key, value in metrics.items():
        run.log(key, value)

    run.upload_file(name="churn_rf.joblib", path_or_stream=model_path)
    model = run.register_model(model_name="churn-rf",
                               model_path=str(model_path),
                               tags={"accuracy": str(metrics['accuracy']),
                                     "f1": str(metrics['f1'])})
    print("Model registered:", model.name)
    run.complete()

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main()
