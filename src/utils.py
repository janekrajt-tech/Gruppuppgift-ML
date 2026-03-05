from __future__ import annotations
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from joblib import dump, load

TARGET_COL = "is_suspicious"
ID_COL = "id"
RANDOM_STATE = 42

# -----------------------------------
# Loading all data (historical and new)
# -----------------------------------

def load_historical(path:str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load historical data (contains target)
    Returns: X, y, df
    """
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataset.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y, df

def load_new(path:str) -> pd.DataFrame:
    """
    Load new_data (no target) when it comes. Safe even if the target column is present, it will be ignored.
    Returns: df
    """
    df = pd.read_csv(path)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    return df

# -----------------------------------
# Preprocessing/Pipeline building functions
# -----------------------------------

def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Infer which columns are categorical and which are numeric based on their dtype.
    - Categorical: dtype == 'object'
    - Numeric: all other dtypes

    """
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols

def build_preprocess(X_schema: pd.DataFrame) -> ColumnTransformer:
    """
    Build a preprocessing:
    - Numeric: median imputation
    - Categorical: most frequent imputation + one-hot encoding
    X_schema is used to "lock" which columns are treated as numeric/categorical, so that if the new data has different dtypes, the pipeline will still work.
    """
    num_cols, cat_cols = infer_feature_types(X_schema)

    preprocess = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore')),
            ]), cat_cols),
        ],
        remainder='drop',  # Drop any columns not specified in transformers
        verbose_feature_names_out=False,  # Keep original column names for easier interpretation
    )
    return preprocess

def make_pipeline(model, X_schema: pd.DataFrame) -> Pipeline:
    """
    Create a pipeline with preprocessing and the given model.
    """
    preprocess = build_preprocess(X_schema)
    pipeline = Pipeline(steps=[
        ('prep', preprocess),
        ('model', model)
    ])
    return pipeline

