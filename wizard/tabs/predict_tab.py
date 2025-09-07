# predict_tab.py
from __future__ import annotations

import gradio as gr
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple
from wizard.state import WizardState

_get_state: Optional[Callable[[], WizardState]] = None

def bind_state(get_state: Callable[[], WizardState]) -> None:
    global _get_state
    _get_state = get_state

NUMERIC_KINDS = ("i", "u", "f")
MAX_FEATS = 30  # pre-allocate controls; toggle visibility per feature

def _nice_step(cmin: float, cmax: float) -> float:
    span = cmax - cmin
    if span <= 0:
        return 0.1
    rough = span / 100.0
    exp = int(np.floor(np.log10(rough))) if rough > 0 else -1
    base = round(rough / (10 ** exp), 1) * (10 ** exp)
    return max(10 ** (exp - 1), base)

def _infer_meta(df: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    for col in features:
        s = df[col]
        if s.dtype.kind in NUMERIC_KINDS:
            cmin = float(np.nanmin(s.values))
            cmax = float(np.nanmax(s.values))
            if not np.isfinite(cmin) or not np.isfinite(cmax):
                cmin, cmax = 0.0, 1.0
            pad = (cmax - cmin) * 0.05 if cmax > cmin else 1.0
            meta[col] = dict(
                kind="numeric",
                min=round(cmin - pad, 6),
                max=round(cmax + pad, 6),
                step=_nice_step(cmin, cmax),
                default=float(np.nanmedian(s.values)),
            )
        else:
            uniq = pd.unique(s.astype(str).fillna("")).tolist()
            uniq = sorted([u for u in uniq if u != ""]) or ["(empty)"]
            meta[col] = dict(kind="categorical", choices=uniq, default=uniq[0])
    return meta

def _predict_one(state: WizardState, values: List[Any]) -> Tuple[str, Optional[pd.DataFrame]]:
    if state is None or state.model is None:
        raise gr.Error("Train a model first in the Train tab.")
    if not state.features:
        raise gr.Error("Select features in the Data/EDA tabs.")

    row = {feat: val for feat, val in zip(state.features, values)}
    X = pd.DataFrame([row], columns=state.features)

    y_pred = state.model.predict(X)
    pred_text = str(y_pred[0])

    proba_df: Optional[pd.DataFrame] = None
    if getattr(state, "task_type", "") == "classification" and hasattr(state.model, "predict_proba"):
        try:
            probs = state.model.predict_proba(X)
            classes = getattr(state.model, "classes_", None)

            # Try to find classes_ inside a Pipeline if not on the top-level
            if classes is None and hasattr(state.model, "named_steps"):
                for step in reversed(getattr(state.model, "named_steps", {}).values()):
                    if hasattr(step, "classes_"):
                        classes = step.classes_
                        break

            if classes is None:
                classes = list(range(probs.shape[1]))

            labels = getattr(state, "class_names", None)
            if labels is not None and len(labels) == len(classes):
                colnames = [str(lbl) for lbl in labels]
            else:
                colnames = [str(c) for c in classes]

            proba_df = (
                pd.DataFrame(probs, columns=colnames)
                .T.rename(columns={0: "probability"})
                .assign(probability=lambda d: d["probability"].astype(float))
                .sort_values("probability", ascending=False)
            )
        except Exception:
            proba_df = None

    return pred_text, proba_df

def make_predict_tab(version_token: gr.Number):
    if _get_state is None:
        raise RuntimeError("bind_state(...) must be called before make_predict_tab().")

    with gr.Tab("Predict"):
        info = gr.Markdown("### 🔮 Predict\nLoad data, pick features, train a model, then predict on custom inputs.")
        warn = gr.Markdown(visible=False)

        # Pre-allocate pairs of controls per feature slot.
        num_inputs: List[gr.Slider] = []
        cat_inputs: List[gr.Dropdown] = []

        with gr.Group():
            with gr.Row():
                for i in range(MAX_FEATS):
                    with gr.Column(scale=1):
                        num_inputs.append(
                            gr.Slider(
                                label=f"num_{i+1}",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=0.0,
                                visible=False,
                            )
                        )
                        cat_inputs.append(
                            gr.Dropdown(
                                label=f"cat_{i+1}",
                                choices=[],
                                value=None,
                                allow_custom_value=True,
                                visible=False,
                            )
                        )

        predict_btn = gr.Button("🧮 Predict", variant="primary")
        y_out = gr.Label(label="Prediction")
        proba_out = gr.Dataframe(
            headers=["class", "probability"],
            row_count=(0, "dynamic"),
            col_count=2,
            wrap=True,
            interactive=False,
            label="Class probabilities (if available)",
        )

        # --- helpers wired to events -------------------------------------------------
        def _rebuild_controls(_token_val):
            st: WizardState = _get_state()
            if st.df_raw is None:
                return (
                    gr.update(value="> Load a dataset first (Data tab)."),
                    gr.update(visible=True, value=""),
                ) + _pack_all_hidden_updates()

            if not st.features:
                return (
                    gr.update(value="> Choose your features in Data/EDA, then train."),
                    gr.update(visible=True, value=""),
                ) + _pack_all_hidden_updates()

            meta = _infer_meta(st.df_raw, st.features)
            upd_num, upd_cat = [], []

            # Build per-slot updates
            for idx in range(MAX_FEATS):
                if idx < len(st.features):
                    feat = st.features[idx]
                    m = meta[feat]
                    if m["kind"] == "numeric":
                        upd_num.append(
                            gr.update(
                                label=feat,
                                minimum=m["min"],
                                maximum=m["max"],
                                step=m["step"],
                                value=m["default"],
                                visible=True,
                            )
                        )
                        upd_cat.append(gr.update(visible=False))
                    else:
                        upd_num.append(gr.update(visible=False))
                        upd_cat.append(
                            gr.update(
                                label=feat,
                                choices=m["choices"],
                                value=m["default"],
                                visible=True,
                            )
                        )
                else:
                    upd_num.append(gr.update(visible=False))
                    upd_cat.append(gr.update(visible=False))

            # Clear outputs and show small status
            msg = f"Ready • {len(st.features)} features • Task: {st.task_type or '?'}"
            return (gr.update(value=f"### 🔮 Predict\n{msg}"),
                    gr.update(visible=False)) + tuple(upd_num + upd_cat) + (
                        gr.update(value=""),
                        gr.update(value=pd.DataFrame(columns=["class", "probability"]))
                    )

        def _pack_all_hidden_updates():
            # Hide all pre-allocated inputs; also clear outputs
            upd_num = [gr.update(visible=False) for _ in range(MAX_FEATS)]
            upd_cat = [gr.update(visible=False) for _ in range(MAX_FEATS)]
            clear_pred = gr.update(value="")
            clear_tbl = gr.update(value=pd.DataFrame(columns=["class", "probability"]))
            return tuple(upd_num + upd_cat + [clear_pred, clear_tbl])

        def _on_predict(*vals):
            st: WizardState = _get_state()
            if st.df_raw is None:
                raise gr.Error("Load a dataset first.")
            if not st.features:
                raise gr.Error("Select features in the Data/EDA tabs.")
            if st.model is None:
                raise gr.Error("Train a model first in the Train tab.")

            # Pull values in feature order from the appropriate control per slot
            picked: List[Any] = []
            for idx, feat in enumerate(st.features):
                # Determine which control is visible at this slot
                num_val = vals[idx]                    # first MAX_FEATS are sliders
                cat_val = vals[MAX_FEATS + idx]        # next MAX_FEATS are dropdowns
                # Use which one is not None
                val = num_val if num_val is not None else cat_val
                picked.append(val)

            pred, probadf = _predict_one(st, picked)
            if probadf is None or probadf.empty:
                return pred, gr.update(value=pd.DataFrame(columns=["class", "probability"]))
            df = probadf.reset_index().rename(columns={"index": "class"})
            return pred, df

        # Wire refresh on data_version token
        version_token.change(
            _rebuild_controls,
            inputs=[version_token],
            outputs=[info, warn] + num_inputs + cat_inputs + [y_out, proba_out],
        )

        # Initial render (handles first load or samples)
        # We reuse the same function with the current token value
        dummy_init = gr.Button(visible=False)
        dummy_init.click(
            _rebuild_controls,
            inputs=[version_token],
            outputs=[info, warn] + num_inputs + cat_inputs + [y_out, proba_out],
        )

        # Predict click wiring
        predict_btn.click(
            _on_predict,
            inputs=num_inputs + cat_inputs,   # order matters for _on_predict
            outputs=[y_out, proba_out],
        )
