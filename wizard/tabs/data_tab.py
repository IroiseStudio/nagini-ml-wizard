# data_tab.py
from __future__ import annotations
import gradio as gr
import pandas as pd
from typing import Callable, Optional
from wizard.state import WizardState
from wizard import data_io

_get_state: Optional[Callable[[], WizardState]] = None

def bind_state(get_state: Callable[[], WizardState]) -> None:
    global _get_state
    _get_state = get_state

def _infer_task(df: pd.DataFrame, target: str) -> str:
    nunq = df[target].nunique(dropna=True) if target in df.columns else 0
    return "classification" if (nunq > 0 and nunq <= 20) else "regression"

def make_data_tab(version_token: gr.Number):
    if _get_state is None:
        raise RuntimeError("bind_state(...) must be called before make_data_tab().")

    with gr.Tab("Data"):
        info = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload CSV")
                load_btn = gr.Button("Load CSV")
            with gr.Column(scale=1):
                sample_dd = gr.Dropdown(
                    ["Wine Quality", "Iris", "Wine", "Diabetes"],
                    label="or load a sample"
                )
                sample_btn = gr.Button("Load sample")

        df_show = gr.Dataframe(interactive=False)

        def on_load_csv(f):
            st: WizardState = _get_state()
            if f is None:
                raise gr.Error("Upload a CSV or choose a sample.")
            df = pd.read_csv(f.name)
            df.columns = [str(c) for c in df.columns]

            target_guess = df.columns[-1]
            features = [c for c in df.columns if c != target_guess]
            task_guess = _infer_task(df, target_guess)

            st.df_raw = df
            st.df_clean = None
            st.target = target_guess
            st.features = features
            st.task_type = task_guess
            st.data_version += 1  # signal

            info_txt = (
                f"Rows {len(df)} | Cols {len(df.columns)} | "
                f"Target guess: {target_guess} | Task: {task_guess} | v{st.data_version}"
            )
            return info_txt, df.head(10), st, gr.update(value=st.data_version)

        def on_load_sample_click(name: str):
            st: WizardState = _get_state()
            df, tgt, task = data_io.load_sample(name)
            df.columns = [str(c) for c in df.columns]
            features = [c for c in df.columns if c != tgt]

            st.df_raw = df
            st.df_clean = None
            st.target = tgt
            st.features = features
            st.task_type = task
            st.data_version += 1  # signal

            info_txt = (
                f"{name} loaded | Rows: {len(df)} | Cols: {len(df.columns)} | "
                f"Target: {tgt} | Task: {task} | v{st.data_version}"
            )
            return info_txt, df.head(10), st, gr.update(value=st.data_version)

        # IMPORTANT: output the shared token (version_token), not a new Number
        load_btn.click(on_load_csv, [file_in], [info, df_show, gr.State(_get_state()), version_token])
        sample_btn.click(on_load_sample_click, [sample_dd], [info, df_show, gr.State(_get_state()), version_token])