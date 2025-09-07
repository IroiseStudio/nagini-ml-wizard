# eda_tab.py
from __future__ import annotations
import gradio as gr
from typing import Callable, Optional
from wizard.state import WizardState
from wizard import preprocess, eda

_get_state: Optional[Callable[[], WizardState]] = None

def bind_state(get_state: Callable[[], WizardState]) -> None:
    global _get_state
    _get_state = get_state

def make_eda_tab(version_token: gr.Number):
    if _get_state is None:
        raise RuntimeError("bind_state(...) must be called before make_eda_tab().")

    with gr.Tab("Preprocess & EDA"):
        target_dd = gr.Dropdown(label="Target")
        feats_cg = gr.CheckboxGroup(label="Features")
        task_txt = gr.Textbox(label="Inferred Task", interactive=False)

        num_missing = gr.Radio(["none", "mean", "median"], value="mean", label="Numeric missing")
        cat_missing = gr.Radio(["none", "mode"], value="mode", label="Categorical missing")
        drop_na = gr.Checkbox(label="Drop rows with any NA")
        scale_num = gr.Checkbox(label="Scale numeric")
        enc_cat = gr.Checkbox(label="Encode categoricals", value=True)
        apply_btn = gr.Button("Apply & Plot")

        class_hist = gr.Image(label="Target distribution", interactive=False, type="pil")
        corr_img  = gr.Image(label="Correlation (numeric)", interactive=False, type="pil")

        plot_gate_msg = gr.Markdown("➡️ **Press _Apply & Plot_ above to enable plotting.**", visible=True)

        feat_plot_dd = gr.Dropdown(label="Feature to plot vs target", interactive=False)
        plot_kind = gr.Radio(["Histogram by class", "Box/Violin"], value="Histogram by class",
                             label="Plot type", interactive=False)
        draw_btn = gr.Button("Plot feature vs target", interactive=False)
        feat_plot = gr.Image(label="Feature vs Target", interactive=False, type="pil")

        def on_data_version_changed(_: float):
            st: WizardState = _get_state()
            df = st.df_raw
            if df is None or st.target is None:
                tgt_upd = gr.update(choices=[], value=None)
                f_upd   = gr.update(choices=[], value=[], interactive=False)
                feat_dd = gr.update(choices=[], value=None, interactive=False)
                gate    = gr.update(visible=True)
                disable = gr.update(interactive=False)
                clear   = None
                return (tgt_upd, f_upd, "", clear, clear, feat_dd, disable, disable, gate, clear)

            tgt_upd = gr.update(choices=df.columns.tolist(), value=st.target)
            f_upd   = gr.update(choices=st.features, value=st.features)
            task    = st.task_type or ""
            feat_dd = gr.update(choices=st.features,
                                value=(st.features[0] if st.features else None),
                                interactive=False)
            gate    = gr.update(visible=True)
            disable = gr.update(interactive=False)
            clear   = None
            return (tgt_upd, f_upd, task, clear, clear, feat_dd, disable, disable, gate, clear)

        # Listen to the shared token
        version_token.change(
            on_data_version_changed,
            [version_token],
            [target_dd, feats_cg, task_txt, class_hist, corr_img, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg, feat_plot],
        )

        def on_prepare(target, feats, nmis, cmis, dropany, scale, enc):
            st: WizardState = _get_state()
            if st.df_raw is None:
                raise gr.Error("Load data first.")
            if target not in st.df_raw.columns:
                raise gr.Error("Target not in data.")
            feats = [c for c in feats if c != target]
            if not feats:
                raise gr.Error("Pick at least one feature.")

            st.target = target
            st.features = feats
            st.num_missing = nmis
            st.cat_missing = cmis
            st.drop_any_na = dropany
            st.scale_numeric = scale
            st.encode_categorical = enc

            df = preprocess.apply_row_drop(st.df_raw, feats + [target], dropany)
            st.df_clean = df
            td = eda.target_distribution(df, target)
            cr = eda.correlation(df[feats + [target]])

            feat_dd_update = gr.update(choices=feats, value=(feats[0] if feats else None), interactive=True,
                                       label=f"Feature to plot vs target (target: **{target}**)")
            enable = gr.update(interactive=True)
            hide_gate = gr.update(visible=False)

            return td, cr, st, feat_dd_update, enable, enable, hide_gate

        apply_btn.click(
            on_prepare,
            [target_dd, feats_cg, num_missing, cat_missing, drop_na, scale_num, enc_cat],
            [class_hist, corr_img, gr.State(_get_state()), feat_plot_dd, plot_kind, draw_btn, plot_gate_msg],
        )

        def on_draw_feature_plot(feature, kind):
            st: WizardState = _get_state()
            df = st.df_clean if st.df_clean is not None else st.df_raw
            if df is None:
                raise gr.Error("Load data first.")
            if feature is None or (st.features and feature not in st.features):
                raise gr.Error("Pick a feature to plot.")
            if st.target is None or st.target not in df.columns:
                raise gr.Error("Please set a valid target.")
            return (eda.feature_hist_by_target(df, feature, st.target)
                    if kind == "Histogram by class"
                    else eda.feature_box_by_target(df, feature, st.target))

        draw_btn.click(on_draw_feature_plot, [feat_plot_dd, plot_kind], [feat_plot])

        # Keep target/features consistent if edited directly
        def on_target_changed(new_target: str):
            st: WizardState = _get_state()
            if st.df_raw is None:
                raise gr.Error("Load data first.")
            st.target = new_target
            valid_feats = [f for f in (st.features or []) if f != new_target]
            st.features = valid_feats
            clear = None
            feat_dd_update = gr.update(choices=valid_feats, value=None, interactive=False,
                                       label=f"Feature to plot vs target (target: **{new_target}**)")
            disable = gr.update(interactive=False)
            gate = gr.update(visible=True)
            return clear, clear, clear, feat_dd_update, disable, disable, gate

        target_dd.change(on_target_changed, [target_dd],
                         [class_hist, corr_img, feat_plot, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg])

        def on_features_changed(new_feats: list[str]):
            st: WizardState = _get_state()
            if st.df_raw is None:
                raise gr.Error("Load data first.")
            cleaned = [f for f in (new_feats or []) if f != st.target]
            st.features = cleaned
            clear = None
            feat_dd_update = gr.update(choices=cleaned, value=None, interactive=False)
            disable = gr.update(interactive=False)
            gate = gr.update(visible=True)
            return clear, feat_dd_update, disable, disable, gate

        feats_cg.change(on_features_changed, [feats_cg],
                        [feat_plot, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg])
