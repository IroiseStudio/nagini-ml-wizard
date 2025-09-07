import gradio as gr
import pandas as pd
from wizard import state
from wizard.state import WizardState
from wizard import data_io, preprocess
from wizard import eda

from wizard.tabs.eda_tab import make_eda_tab, bind_state as bind_state_eda
from wizard.tabs.data_tab import make_data_tab, bind_state as bind_state_data
from wizard.tabs.train_tab import make_train_tab, bind_state as bind_state_train
from wizard.tabs.predict_tab import make_predict_tab, bind_state as bind_predict_tab 


with gr.Blocks(title="🐍🧙 Nagini ML Wizard") as demo:
    st = gr.State(WizardState())

    # Give all tabs access to the same WizardState object
    get_state = lambda: st.value
    bind_state_data(get_state)
    bind_state_eda(get_state)
    bind_state_train(get_state)
    bind_predict_tab(get_state)    

    gr.Markdown(
        """
        # 🐍🧙 Nagini ML Wizard  
        Small, clear steps from **data → EDA → train → evaluate → predict**.

        This wizard guides you interactively through the *full ML workflow*:
        - Import your dataset  
        - Explore with EDA plots  
        - Preprocess and guard against leakage  
        - Train models with configurable hyperparameters  
        - Evaluate with the right metrics  
        - Make single or batch predictions  
        - Export your trained pipeline  
        """
    )

    data_version_token = gr.Number(value=0, visible=False)

    # 1) Data tab
    make_data_tab(version_token=data_version_token)

    # 2) EDA tab
    make_eda_tab(version_token=data_version_token)

    # 3) Train tab
    make_train_tab()

    # 4) Predict tab
    make_predict_tab(version_token=data_version_token)

    # -------------------------
    # Tab 1: Data (UPLOAD ONLY)
    # -------------------------
    # with gr.Tab("Data"):
    #     info = gr.Markdown()

    #     with gr.Row():
    #         with gr.Column(scale=1):
    #             file_in = gr.File(label="Upload CSV")
    #             load_btn = gr.Button("Load CSV")  # under file input
    #         with gr.Column(scale=1):
    #             sample_dd = gr.Dropdown(
    #                 ["Wine Quality", "Iris", "Wine", "Diabetes"],
    #                 label="or load a sample"
    #             )
    #             sample_btn = gr.Button("Load sample")  # under dropdown

    #     df_show = gr.Dataframe(interactive=False)

    # # -----------------------------------------
    # # Tab 2: Preprocess & EDA (ALL CONTROLS)
    # # -----------------------------------------
    # with gr.Tab("Preprocess & EDA"):
    #     # Selection controls live here (moved out of Data tab)
    #     target_dd = gr.Dropdown(label="Target")
    #     feats_cg = gr.CheckboxGroup(label="Features")

    #     # Preprocess options
    #     num_missing = gr.Radio(["none","mean","median"], value="mean", label="Numeric missing")
    #     cat_missing = gr.Radio(["none","mode"], value="mode", label="Categorical missing")
    #     drop_na = gr.Checkbox(label="Drop rows with any NA")
    #     scale_num = gr.Checkbox(label="Scale numeric")
    #     enc_cat = gr.Checkbox(label="Encode categoricals", value=True)
    #     apply_btn = gr.Button("Apply & Plot")

    #     # EDA outputs
    #     class_hist = gr.Image(label="Target distribution", interactive=False, type="pil")
    #     corr_img  = gr.Image(label="Correlation (numeric)", interactive=False, type="pil")

    #     # helper message shown until Apply & Plot runs
    #     plot_gate_msg = gr.Markdown("➡️ **Press _Apply & Plot_ above to enable plotting.**", visible=True)

    #     # Feature-vs-target plot controls (start disabled)
    #     feat_plot_dd = gr.Dropdown(label="Feature to plot vs target", interactive=False)
    #     plot_kind = gr.Radio(
    #         ["Histogram by class", "Box/Violin"],
    #         value="Histogram by class",
    #         label="Plot type",
    #         interactive=False
    #     )
    #     draw_btn = gr.Button("Plot feature vs target", interactive=False)
    #     feat_plot = gr.Image(label="Feature vs Target", interactive=False, type="pil")

    #     # Optional: inferred task display for user clarity
    #     task_txt = gr.Textbox(label="Inferred Task", interactive=False)

    #     # ---- callbacks for EDA/preprocess ----
    #     def on_prepare(target, feats, nmis, cmis, dropany, scale, enc, state: WizardState):
    #         if state.df_raw is None:
    #             raise gr.Error("Load data first.")
    #         if target not in state.df_raw.columns:
    #             raise gr.Error("Target not in data.")
    #         feats = [c for c in feats if c != target]
    #         if not feats:
    #             raise gr.Error("Pick at least one feature.")

    #         state.target = target
    #         state.features = feats
    #         state.num_missing = nmis
    #         state.cat_missing = cmis
    #         state.drop_any_na = dropany
    #         state.scale_numeric = scale
    #         state.encode_categorical = enc

    #         df = preprocess.apply_row_drop(state.df_raw, feats + [target], dropany)
    #         state.df_clean = df
    #         td = eda.target_distribution(df, target)
    #         cr = eda.correlation(df[feats + [target]])

    #         # build updates
    #         feat_dd_update = gr.update(
    #             choices=feats,
    #             value=(feats[0] if feats else None),
    #             interactive=True,
    #             label=f"Feature to plot vs target (target: **{target}**)"
    #         )
    #         plot_kind_update = gr.update(interactive=True)
    #         draw_btn_update = gr.update(interactive=True)
    #         gate_hide = gr.update(visible=False)

    #         return (
    #             td,
    #             cr,
    #             state,
    #             feat_dd_update,
    #             plot_kind_update,
    #             draw_btn_update,
    #             gate_hide,
    #         )

    #     apply_btn.click(
    #         on_prepare,
    #         [target_dd, feats_cg, num_missing, cat_missing, drop_na, scale_num, enc_cat, st],
    #         [class_hist, corr_img, st, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg],
    #     )

    #     def on_draw_feature_plot(feature, kind, state: WizardState):
    #         # Allow plotting immediately after load: fall back to raw df
    #         df = state.df_clean if state.df_clean is not None else state.df_raw
    #         if df is None:
    #             raise gr.Error("Load data first.")
    #         if feature is None or (state.features and feature not in state.features):
    #             raise gr.Error("Pick a feature to plot.")
    #         if state.target is None or state.target not in df.columns:
    #             raise gr.Error("Please set a valid target.")
    #         if kind == "Histogram by class":
    #             return eda.feature_hist_by_target(df, feature, state.target)
    #         else:
    #             return eda.feature_box_by_target(df, feature, state.target)

    #     draw_btn.click(on_draw_feature_plot, [feat_plot_dd, plot_kind, st], [feat_plot])


    #     def on_target_changed(new_target: str, state: WizardState):
    #         if state.df_raw is None:
    #             raise gr.Error("Load data first.")

    #         # Update state with the new target
    #         state.target = new_target

    #         # Keep features valid (no target leakage)
    #         valid_feats = [f for f in (state.features or []) if f != new_target]
    #         state.features = valid_feats

    #         # Build updates:
    #         #   - clear images
    #         #   - disable plotting controls
    #         #   - relabel feature dropdown to include target
    #         #   - show the "Apply & Plot" gate message
    #         class_hist_clear = None
    #         corr_img_clear   = None
    #         feat_plot_clear  = None

    #         feat_dd_update = gr.update(
    #             choices=valid_feats,
    #             value=None,
    #             interactive=False,
    #             label=f"Feature to plot vs target (target: **{new_target}**)"
    #         )
    #         plot_kind_disable = gr.update(interactive=False)
    #         draw_btn_disable  = gr.update(interactive=False)
    #         gate_show         = gr.update(visible=True)

    #         return (class_hist_clear, corr_img_clear, feat_plot_clear,
    #                 feat_dd_update, plot_kind_disable, draw_btn_disable, gate_show, state)
        
    #     target_dd.change(
    #         on_target_changed,
    #         [target_dd, st],
    #         [class_hist, corr_img, feat_plot, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg, st],
    #     )

    #     def on_features_changed(new_feats: list[str], state: WizardState):
    #         if state.df_raw is None:
    #             raise gr.Error("Load data first.")
    #         # enforce no-target in features
    #         cleaned = [f for f in (new_feats or []) if f != state.target]
    #         state.features = cleaned

    #         # Clear only the feature-vs-target image and disable plot until Apply & Plot,
    #         # OR keep it enabled if you prefer. Here we disable for consistency.
    #         feat_plot_clear   = None  # <-- this clears the feat_plot Image
    #         feat_dd_update = gr.update(choices=cleaned, value=None, interactive=False)
    #         plot_kind_disable = gr.update(interactive=False)
    #         draw_btn_disable  = gr.update(interactive=False)
    #         gate_show         = gr.update(visible=True)

    #         return feat_plot_clear, feat_dd_update, plot_kind_disable, draw_btn_disable, gate_show, state

    #     feats_cg.change(
    #         on_features_changed,
    #         [feats_cg, st],
    #         [feat_plot, feat_plot_dd, plot_kind, draw_btn, plot_gate_msg, st],
    #     )

    # # # ----------------------------------------------------
    # # # Define load helpers AFTER tabs so components exist
    # # # ----------------------------------------------------
    # def _infer_task(df: pd.DataFrame, target: str) -> str:
    #     nunq = df[target].nunique(dropna=True) if target in df.columns else 0
    #     return "classification" if (nunq > 0 and nunq <= 20) else "regression"

    # def on_load_csv(f, state: WizardState):
    #     if f is None:
    #         raise gr.Error("Upload a CSV or choose a sample.")
    #     df = pd.read_csv(f.name)
    #     df.columns = [str(c) for c in df.columns]

    #     target_guess = df.columns[-1]
    #     features = [c for c in df.columns if c != target_guess]
    #     task_guess = _infer_task(df, target_guess)

    #     state.df_raw = df
    #     state.target = target_guess
    #     state.features = features
    #     state.task_type = task_guess

    #     info_txt = f"Rows {len(df)} | Cols {len(df.columns)} | Target guess: {target_guess} | Task: {task_guess}"
    #     tgt_update   = gr.update(choices=df.columns.tolist(), value=target_guess)
    #     feats_update = gr.update(choices=features, value=features)
    #     feat_plot_update = gr.update(choices=features, value=(features[0] if features else None))
    #     return info_txt, df.head(10), state, tgt_update, feats_update, feat_plot_update, task_guess

    # def on_load_sample_click(name: str, state: WizardState):
    #     df, tgt, task = data_io.load_sample(name)  # supports "Wine Quality"
    #     df.columns = [str(c) for c in df.columns]
    #     features = [c for c in df.columns if c != tgt]

    #     state.df_raw = df
    #     state.target = tgt
    #     state.features = features
    #     state.task_type = task

    #     info_txt = f"{name} loaded | Rows: {len(df)} | Cols: {len(df.columns)} | Target: {tgt} | Task: {task}"
    #     tgt_update   = gr.update(choices=df.columns.tolist(), value=tgt)
    #     feats_update = gr.update(choices=features, value=features)
    #     feat_plot_update = gr.update(choices=features, value=(features[0] if features else None))
    #     return info_txt, df.head(10), state, tgt_update, feats_update, feat_plot_update, task

    # # Bind after both tabs so target/feature widgets exist
    # load_btn.click(
    #     on_load_csv,
    #     [file_in, st],
    #     [info, df_show, st, target_dd, feats_cg, feat_plot_dd, task_txt],
    # )
    # sample_btn.click(
    #     on_load_sample_click,
    #     [sample_dd, st],
    #     [info, df_show, st, target_dd, feats_cg, feat_plot_dd, task_txt],
    # )

    # with gr.Tab("Train"):
    #     make_train_tab()

    # Predict tab would read state.pipeline and auto-build inputs…
demo.launch()
