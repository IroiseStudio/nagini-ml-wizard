from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

def split_xy(df, features, target):
    X = df[features]
    y = df[target]
    return X, y

def train_pipeline(preprocessor, estimator, X, y, task, test_size=0.2, random_state=42, stratify=True) -> Tuple[Pipeline, Dict[str, Any], plt.Figure]:
    strat = y if (task == "classification" and stratify and len(np.unique(y)) > 1) else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

    pipe = Pipeline([("pre", preprocessor), ("model", estimator)])
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)

    if task == "classification":
        acc = accuracy_score(y_te, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
        report = classification_report(y_te, y_pred, zero_division=0)
        cm = confusion_matrix(y_te, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(im)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        metrics = {"accuracy": acc, "precision_w": pr, "recall_w": rc, "f1_w": f1, "report": report, "cm": cm}
    else:
        mae = mean_absolute_error(y_te, y_pred)
        rmse = mean_squared_error(y_te, y_pred, squared=False)
        r2 = r2_score(y_te, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_te, y_pred, alpha=0.7)
        ax.set_title("y_true vs y_pred")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    fig.tight_layout()
    return pipe, metrics, fig
