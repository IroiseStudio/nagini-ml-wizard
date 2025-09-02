from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

def make_preprocessor(
    df: pd.DataFrame,
    features: List[str],
    num_missing: str,
    cat_missing: str,
    scale_numeric: bool,
    encode_categorical: bool,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df[features]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features if c not in num_cols]

    num_steps = []
    if num_missing in {"mean", "median"}:
        strategy = "mean" if num_missing == "mean" else "median"
        num_steps.append(("imputer", SimpleImputer(strategy=strategy)))
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = []
    if cat_missing == "mode":
        cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
    if encode_categorical and cat_cols:
        cat_steps.append(("ohe", OneHotEncoder(handle_unknown="ignore")))
    # If not encoding, categorical pass-through

    transformers = []
    if num_cols:
        transformers.append(("num", make_pipeline(*num_steps) if num_steps else "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", make_pipeline(*cat_steps) if cat_steps else "passthrough", cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, cat_cols

def apply_row_drop(df: pd.DataFrame, cols: List[str], drop_any_na: bool) -> pd.DataFrame:
    if drop_any_na:
        return df.dropna(subset=cols, how="any").copy()
    return df.copy()
