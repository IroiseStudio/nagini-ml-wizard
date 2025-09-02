# wizard/eda.py
import io
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional
import numpy as np
from PIL import Image

def fig_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    img.load()  # fully read into memory so buf can be freed
    return img

def target_distribution(df: pd.DataFrame, target: str) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    s = df[target].dropna()
    if s.nunique() <= 20:
        s.value_counts().sort_index().plot(kind="bar", ax=ax)
        ax.set_title("Class Balance")
        ax.set_xlabel("Class"); ax.set_ylabel("Count")
    else:
        ax.hist(s.values, bins=30)
        ax.set_title("Target Distribution")
        ax.set_xlabel(target); ax.set_ylabel("Frequency")
    return fig_pil(fig)

def correlation(df: pd.DataFrame) -> Image.Image:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr.values, interpolation="nearest")
    fig.colorbar(im)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    return fig_pil(fig)

def feature_hist_by_target(
    df: pd.DataFrame, feature: str, target: str, bins: int = 30
) -> Image.Image:
    """
    Overlay histograms of `feature` for each target class (classification).
    If target is numeric with many unique values (regression), fall back to scatter.
    """
    s_target = df[target].dropna()
    # classification heuristic
    is_class = s_target.nunique(dropna=True) <= 20 and not np.issubdtype(s_target.dtype, np.floating)

    fig, ax = plt.subplots(figsize=(6, 4))
    if is_class:
        for cls, grp in df[[feature, target]].dropna().groupby(target):
            ax.hist(grp[feature].values, bins=bins, alpha=0.5, label=str(cls))
        ax.legend(title=target, bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_title(f"{feature} by {target} (histograms)")
        ax.set_xlabel(feature); ax.set_ylabel("Frequency")
    else:
        # regression fallback: scatter
        xy = df[[feature, target]].dropna()
        ax.scatter(xy[feature].values, xy[target].values, alpha=0.7)
        ax.set_title(f"{feature} vs {target} (scatter)")
        ax.set_xlabel(feature); ax.set_ylabel(target)
    return fig_pil(fig)

def feature_box_by_target(df: pd.DataFrame, feature: str, target: str) -> Image.Image:
    """
    Boxplot of `feature` grouped by target classes (classification).
    If regression target, show violin of feature binned by target quantiles.
    """
    s_target = df[target].dropna()
    is_class = s_target.nunique(dropna=True) <= 20 and not np.issubdtype(s_target.dtype, np.floating)

    fig, ax = plt.subplots(figsize=(6, 4))
    if is_class:
        df[[feature, target]].dropna().boxplot(column=feature, by=target, ax=ax)
        ax.set_title(f"{feature} by {target} (boxplot)"); ax.set_xlabel(target); ax.set_ylabel(feature)
        fig.suptitle("")
    else:
        # bin target into quantiles to visualize distribution of feature
        tmp = df[[feature, target]].dropna().copy()
        tmp["_tbin"] = pd.qcut(tmp[target], q=min(6, max(2, int(np.sqrt(len(tmp))//50) or 4)), duplicates="drop")
        groups = [g[feature].values for _, g in tmp.groupby("_tbin")]
        ax.violinplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([str(c) for c in tmp["_tbin"].cat.categories], rotation=45, ha="right")
        ax.set_title(f"{feature} across target bins (violin)"); ax.set_xlabel("Target bins"); ax.set_ylabel(feature)
    return fig_pil(fig)