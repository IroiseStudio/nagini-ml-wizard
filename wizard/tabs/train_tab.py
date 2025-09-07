
"""
wizard/tabs/train_tab.py

Train Tab (WizardState compatible)
----------------------------------
UI sections:
1) Model Select
2) Parameters
3) Train & Results

Assumes a global WizardState-like object with:
- df_clean (preferred) or df_raw
- target, features, task_type ("classification"|"regression")
- test_size, random_state, stratify
- pipeline, metrics, model_name

Usage:
    from wizard.tabs.train_tab import make_train_tab, bind_state
    bind_state(lambda: state)  # provide access to your WizardState
    with gr.Tab("Train"):
        make_train_tab()
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, root_mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance

# Estimators
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Model registry ----------------

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "Decision Tree": {
        "key": "DT",
        "tasks": ["classification", "regression"],
        "params": ["max_depth", "min_samples_split", "min_samples_leaf"],
        "defaults": {"max_depth": None, "min_samples_split": 2, "min_samples_leaf": 1},
    },
    "Random Forest": {
        "key": "RF",
        "tasks": ["classification", "regression"],
        "params": ["n_estimators", "max_depth", "min_samples_split"],
        "defaults": {"n_estimators": 200, "max_depth": None, "min_samples_split": 2},
    },
    "KNN": {
        "key": "KNN",
        "tasks": ["classification", "regression"],
        "params": ["n_neighbors", "weights"],
        "defaults": {"n_neighbors": 5, "weights": "uniform"},
    },
    "SVM": {
        "key": "SVM",
        "tasks": ["classification", "regression"],
        "params": ["C", "kernel", "gamma", "epsilon", "degree"],
        "defaults": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "epsilon": 0.1, "degree": 3},
    },
    "MLP (Neural Net)": {
        "key": "MLP",
        "tasks": ["classification", "regression"],
        "params": ["hidden_layer_sizes", "activation", "alpha", "max_iter"],
        "defaults": {"hidden_layer_sizes": "64,64", "activation": "relu", "alpha": 1e-4, "max_iter": 500},
    },
    "Linear/Logistic Regression": {
        "key": "LR",
        "tasks": ["classification", "regression"],
        "params": ["C", "fit_intercept"],
        "defaults": {"C": 1.0, "fit_intercept": True},
    },
    "Naive Bayes (Gaussian)": {
        "key": "NB",
        "tasks": ["classification"],
        "params": [],
        "defaults": {},
    },
}


def _parse_hidden(s: str) -> tuple[int, ...]:
    """
    Parse '64,64' → (64, 64). Any non-positive or invalid numbers are fixed.
    Falls back to (64, 64) if empty/bad.
    """
    try:
        tokens = [p.strip() for p in str(s or "").split(",") if p.strip()]
        parts = [max(1, int(float(p))) for p in tokens]  # tolerate '64.0'
        return tuple(parts) if parts else (64, 64)
    except Exception:
        return (64, 64)


def _as_float(x, default):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

def _as_int(x, default):
    try:
        if x is None or x == "": return default
        return int(float(x))  # tolerate "5.0"
    except Exception:
        return default

def _sanitize_params(model_name: str, raw: dict) -> dict:
    # start from defaults; overlay raw
    params = {**MODEL_SPECS[model_name]["defaults"], **(raw or {})}
    key = MODEL_SPECS[model_name]["key"]

    if key in ("DT", "RF"):
        md = params.get("max_depth", None)
        if isinstance(md, (int, float)) and md <= 0:
            md = None
        params["max_depth"] = md

        mss = _as_int(params.get("min_samples_split", 2), 2)
        params["min_samples_split"] = max(2, mss)

        msl = _as_int(params.get("min_samples_leaf", 1), 1)
        params["min_samples_leaf"] = max(1, msl)

        if key == "RF":
            ne = _as_int(params.get("n_estimators", 200), 200)
            params["n_estimators"] = max(10, ne)

    elif key == "KNN":
        n = _as_int(params.get("n_neighbors", 5), 5)
        params["n_neighbors"] = max(1, n)
        w = str(params.get("weights", "uniform"))
        params["weights"] = w if w in ("uniform", "distance") else "uniform"

    elif key == "SVM":
        C = _as_float(params.get("C", 1.0), 1.0)
        params["C"] = max(1e-9, C)

        eps = _as_float(params.get("epsilon", 0.1), 0.1)
        params["epsilon"] = max(1e-9, eps)

        deg = _as_int(params.get("degree", 3), 3)
        params["degree"] = max(1, deg)

        ker = str(params.get("kernel", "rbf"))
        params["kernel"] = ker if ker in ("linear", "rbf", "poly", "sigmoid") else "rbf"

        g = params.get("gamma", "scale")
        if isinstance(g, str):
            g2 = g.strip().lower()
            params["gamma"] = g2 if g2 in ("scale", "auto") else "scale"
        else:
            gf = _as_float(g, 0.1)
            params["gamma"] = max(1e-9, gf)

    elif key == "MLP":
        params["hidden_layer_sizes"] = str(params.get("hidden_layer_sizes", "64,64"))
        a = _as_float(params.get("alpha", 1e-4), 1e-4)
        params["alpha"] = max(1e-12, a)
        mi = _as_int(params.get("max_iter", 500), 500)
        params["max_iter"] = max(50, mi)
        act = str(params.get("activation", "relu"))
        params["activation"] = act if act in ("relu", "tanh", "logistic") else "relu"

    elif key == "LR":
        C = _as_float(params.get("C", 1.0), 1.0)
        params["C"] = max(1e-9, C)
        params["fit_intercept"] = bool(params.get("fit_intercept", True))

    # NB: no params to sanitize
    return params


def _build_estimator(model_name: str, task: str, params: Dict[str, Any]):
    k = MODEL_SPECS[model_name]["key"]
    if k == "DT":
        if task == "classification":
            return DecisionTreeClassifier(
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                min_samples_leaf=params.get("min_samples_leaf", 1),
                random_state=42,
            )
        return DecisionTreeRegressor(
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=42,
        )
    if k == "RF":
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth"),
                min_samples_split=params.get("min_samples_split", 2),
                n_jobs=-1,
                random_state=42,
            )
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            n_jobs=-1,
            random_state=42,
        )
    if k == "KNN":
        if task == "classification":
            return KNeighborsClassifier(
                n_neighbors=params.get("n_neighbors", 5),
                weights=params.get("weights", "uniform"),
                n_jobs=-1,
            )
        return KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "uniform"),
            n_jobs=-1,
        )
    if k == "SVM":
        C = max(1e-9, float(params.get("C", 1.0)))
        kernel = params.get("kernel", "rbf")
        gamma = params.get("gamma", "scale")
        degree = max(1, int(params.get("degree", 3)))  # clamp >=1
        eps = max(1e-9, float(params.get("epsilon", 0.1)))

        if task == "classification":
            return SVC(
                C=C, kernel=kernel, gamma=gamma,
                degree=degree,
                probability=True, random_state=42
            )
        return SVR(
            C=C, kernel=kernel, gamma=gamma,
            degree=degree, epsilon=eps
        )
    if k == "MLP":
        hidden_raw = params.get("hidden_layer_sizes", "64,64")
        if isinstance(hidden_raw, str):
            hidden = _parse_hidden(hidden_raw)
        elif isinstance(hidden_raw, (list, tuple)):
            # Clamp each layer size to >=1; fallback if any conversion fails
            try:
                hidden = tuple(max(1, int(float(x))) for x in hidden_raw)
            except Exception:
                hidden = (64, 64)
        else:
            hidden = (64, 64)

        if task == "classification":
            return MLPClassifier(
                hidden_layer_sizes=hidden,
                activation=params.get("activation", "relu"),
                alpha=float(params.get("alpha", 1e-4)),
                max_iter=int(params.get("max_iter", 500)),
                random_state=42,
            )
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=params.get("activation", "relu"),
            alpha=float(params.get("alpha", 1e-4)),
            max_iter=int(params.get("max_iter", 500)),
            random_state=42,
        )
    if k == "LR":
        if task == "classification":
            return LogisticRegression(C=params.get("C", 1.0), max_iter=200, n_jobs=-1, random_state=42)
        return LinearRegression(fit_intercept=bool(params.get("fit_intercept", True)))
    if k == "NB":
        return GaussianNB()

    raise ValueError(f"Unknown model: {model_name}")


def _get_state():
    return make_train_tab._state_getter() if callable(make_train_tab._state_getter) else None


def _grab_training_data():
    st = _get_state()
    if st is None or (st.df_clean is None and st.df_raw is None):
        raise gr.Error("Load data first in the Data tab.")
    df = st.df_clean if st.df_clean is not None else st.df_raw.copy()

    if not st.target or st.target not in df.columns:
        raise gr.Error("Please set a valid target in the Data tab.")

    feats = st.features or [c for c in df.columns if c != st.target]
    task = st.task_type or "classification"

    return st, df, feats, st.target, task


def _build_preprocessor(df: pd.DataFrame, features: List[str], standardize: bool):
    X = df[features]
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize and num_cols:
        num_steps.append(("scaler", StandardScaler()))

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), num_cols),
            ("cat", Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
        ]
    )
    return pre, num_cols, cat_cols


def _metrics_md(task: str, y_true, y_pred) -> str:
    if task == "classification":
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        return f"**Accuracy:** {acc:.4f}  \n**Precision (weighted):** {prec:.4f}  \n**Recall (weighted):** {rec:.4f}  \n**F1 (weighted):** {f1:.4f}"
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return f"**MAE:** {mae:.4f}  \n**RMSE:** {rmse:.4f}  \n**R²:** {r2:.4f}"


def _plot_confmat(y_true, y_pred):
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=110)
    im = ax.imshow(cm, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def _permutation_importance(pipe: Pipeline, X_val, y_val, task: str, max_feats: int = 20):
    try:
        result = permutation_importance(
            pipe, X_val, y_val,
            n_repeats=5, random_state=42, n_jobs=-1,
            scoring="accuracy" if task == "classification" else "r2"
        )
    except Exception:
        return None

    imp = result.importances_mean
    if imp is None or len(imp) == 0:
        return None

    # Get feature names from the ColumnTransformer step
    try:
        prep = pipe.named_steps["prep"]
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"f{i}" for i in range(len(imp))]

    idx = np.argsort(np.abs(imp))[-min(max_feats, len(imp)) :][::-1]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=110)
    ax.barh([feature_names[i] for i in idx][::-1], imp[idx][::-1])
    ax.set_title("Permutation Importance (top features)")
    ax.set_xlabel("Mean importance")
    plt.tight_layout()
    return fig


def _train(model_name: str, param_json: Dict[str, Any], standardize: bool):
    st, df, feats, target, task = _grab_training_data()

    # Split config from state
    test_size = float(getattr(st, "test_size", 0.2) or 0.2)
    random_state = int(getattr(st, "random_state", 42) or 42)
    do_strat = bool(getattr(st, "stratify", True))
    strat = (df[target] if (task == "classification" and do_strat) else None)

    X = df[feats]
    y = df[target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # Preprocess + estimator
    prep, _, _ = _build_preprocessor(df, feats, standardize=standardize)
    # Merge defaults with JSON params
    safe = _sanitize_params(model_name, param_json)
    est = _build_estimator(model_name, task, safe)

    pipe = Pipeline([("prep", prep), ("model", est)])
    pipe.fit(X_tr, y_tr)

    preds = pipe.predict(X_te)
    metrics_md = _metrics_md(task, y_te, preds)
    cm_fig = _plot_confmat(y_te, preds) if task == "classification" else None
    imp_fig = _permutation_importance(pipe, X_te, y_te, task)

    # write back to state
    st.pipeline = pipe
    st.metrics = {"text": metrics_md}
    st.model_name = MODEL_SPECS[model_name]["key"]

    return metrics_md, cm_fig, imp_fig


# ---------------- Gradio UI ----------------

def make_train_tab():
    with gr.Tab("Train"):
        with gr.Accordion("1) Model Select", open=True):
            model = gr.Dropdown(list(MODEL_SPECS.keys()), value="Decision Tree", label="Model")

        with gr.Accordion("2) Parameters", open=True):
            with gr.Row():
                standardize = gr.Checkbox(value=True, label="Standardize numeric features")
                param_json = gr.JSON(value={}, label="Advanced parameters (JSON)", visible=False)

            # small helpers per model
            with gr.Group() as group_dt:
                gr.Markdown("**Decision Tree / Random Forest**")
                # 0 means None (unlimited); sanitizer will map <=0 -> None anyway
                dt_max_depth = gr.Slider(0, 64, value=0, step=1, label="max_depth (0 = None)")
                dt_min_samples_split = gr.Slider(2, 100, value=2, step=1, label="min_samples_split")
                dt_min_samples_leaf = gr.Slider(1, 100, value=1, step=1, label="min_samples_leaf")
                rf_n_estimators = gr.Slider(10, 1000, value=200, step=10, label="n_estimators (RF)")

            with gr.Group(visible=False) as group_knn:
                gr.Markdown("**KNN**")
                knn_n = gr.Slider(1, 500, value=5, step=1, label="n_neighbors")
                knn_weights = gr.Dropdown(["uniform", "distance"], value="uniform", label="weights")

            with gr.Group(visible=False) as group_svm:
                gr.Markdown("**SVM**")
                svm_C = gr.Slider(0.001, 100.0, value=1.0, step=0.001, label="C")
                svm_kernel = gr.Dropdown(["linear", "rbf", "poly", "sigmoid"], value="rbf", label="kernel")
                svm_gamma = gr.Dropdown(["scale", "auto"], value="scale", label="gamma (rbf/poly/sigmoid)")
                svm_degree = gr.Slider(2, 6, value=3, step=1, label="degree (poly only)")
                svm_epsilon = gr.Slider(0.001, 2.0, value=0.1, step=0.001, label="epsilon (SVR only)")

            with gr.Group(visible=False) as group_mlp:
                gr.Markdown("**MLP**")
                mlp_hidden = gr.Textbox(value="64,64", label="hidden_layer_sizes ('64,64' etc.)")
                mlp_activation = gr.Dropdown(["relu", "tanh", "logistic"], value="relu", label="activation")
                mlp_alpha = gr.Slider(1e-6, 1e-1, value=1e-4, step=1e-6, label="alpha (L2)")
                mlp_max_iter = gr.Slider(50, 2000, value=500, step=50, label="max_iter")

            with gr.Group(visible=False) as group_lr:
                gr.Markdown("**Linear/Logistic Regression**")
                lr_C = gr.Slider(0.001, 100.0, value=1.0, step=0.001, label="C (classification)")
                lr_fit_intercept = gr.Checkbox(value=True, label="fit_intercept (regression)")

            with gr.Group(visible=False) as group_nb:
                gr.Markdown("**Naive Bayes (Gaussian)**")
                gr.Markdown("No additional parameters.")

        with gr.Accordion("3) Train & Results", open=True):
            train_btn = gr.Button("Train", variant="primary")
            metrics_out = gr.Markdown(label="Metrics")
            cm_plot = gr.Plot(label="Confusion Matrix")
            imp_plot = gr.Plot(label="Permutation Importance (approx.)")

    # visibility toggles on model change
    def _toggle(model_name: str):
        key = MODEL_SPECS[model_name]["key"]
        return (
            gr.update(visible=key in ("DT", "RF")),
            gr.update(visible=key == "KNN"),
            gr.update(visible=key == "SVM"),
            gr.update(visible=key == "MLP"),
            gr.update(visible=key == "LR"),
            gr.update(visible=key == "NB"),
            gr.update(value={}),  # reset JSON on change
        )

    model.change(
        _toggle,
        inputs=[model],
        outputs=[group_dt, group_knn, group_svm, group_mlp, group_lr, group_nb, param_json],
    )

    # 1) Inputs passed to _merge (includes param_json so we can merge manual edits)
    _inputs_for_merge = [
        model, param_json,
        dt_max_depth, dt_min_samples_split, dt_min_samples_leaf, rf_n_estimators,
        knn_n, knn_weights,
        svm_C, svm_kernel, svm_gamma, svm_epsilon,
        mlp_hidden, mlp_activation, mlp_alpha, mlp_max_iter,
        lr_C, lr_fit_intercept,
    ]

    # 2) Controls that should TRIGGER _merge (exclude param_json to avoid loop)
    _trigger_controls = [
        model,
        dt_max_depth, dt_min_samples_split, dt_min_samples_leaf, rf_n_estimators,
        knn_n, knn_weights,
        svm_C, svm_kernel, svm_gamma, svm_epsilon,
        mlp_hidden, mlp_activation, mlp_alpha, mlp_max_iter,
        lr_C, lr_fit_intercept,
    ]

    # merge helper controls into JSON
    def _merge(model_name: str, param_json_val: Dict[str, Any], *vals):
        params = dict(param_json_val or {})
        key = MODEL_SPECS[model_name]["key"]
        if key in ("DT", "RF"):
            md, mss, msl, n_est = vals[:4]
            if md is not None and md <= 0:
                params["max_depth"] = None
            else:
                params["max_depth"] = int(md) if md is not None else None
            if mss is not None: 
                params["min_samples_split"] = int(mss)
            if msl is not None: 
                params["min_samples_leaf"] = int(msl)
            if key == "RF" and n_est is not None:
                params["n_estimators"] = int(n_est)
        elif key == "KNN":
            n, w = vals[:2]
            if n is not None: 
                params["n_neighbors"] = int(n)
            if w: 
                params["weights"] = str(w)
        elif key == "SVM":
            C, kernel, gamma, eps, deg = vals[:5]
            if C is not None: 
                params["C"] = float(C)
            if kernel: 
                params["kernel"] = str(kernel)
            if gamma:
                try: 
                    params["gamma"] = float(gamma)
                except Exception: 
                    params["gamma"] = gamma
            if eps is not None: 
                params["epsilon"] = float(eps)
            if deg is not None: 
                params["degree"] = int(deg)
        elif key == "MLP":
            s = str(params.get("hidden_layer_sizes", "64,64"))
            fixed = _parse_hidden(s)  # clamps each layer to >=1 and defaults if empty/bad
            params["hidden_layer_sizes"] = ",".join(str(x) for x in fixed)

            a = _as_float(params.get("alpha", 1e-4), 1e-4)
            params["alpha"] = max(1e-12, a)

            mi = _as_int(params.get("max_iter", 500), 500)
            params["max_iter"] = max(50, mi)

            act = str(params.get("activation", "relu"))
            params["activation"] = act if act in ("relu", "tanh", "logistic") else "relu"
        elif key == "LR":
            C, fit_intercept = vals[:2]
            if C is not None: 
                params["C"] = float(C)
            params["fit_intercept"] = bool(fit_intercept)
        return params

    for ctrl in _trigger_controls:
        ctrl.change(_merge, inputs=_inputs_for_merge, outputs=param_json)

    def _on_train(model_name: str, params: Dict[str, Any], stdz: bool):
        return _train(model_name, params, stdz)

    train_btn.click(_on_train, inputs=[model, param_json, standardize], outputs=[metrics_out, cm_plot, imp_plot])


# global state getter binder
def bind_state(getter: Callable[[], Any]):
    make_train_tab._state_getter = getter  # type: ignore


# init placeholder
make_train_tab._state_getter = lambda: None  # type: ignore