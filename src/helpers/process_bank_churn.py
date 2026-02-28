from typing import List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessedData:
    """
    Container for preprocessed dataset and fitted transformers.
    """
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    input_cols: List[str]
    scaler: Optional[StandardScaler]
    encoder: OneHotEncoder


def split_data(
    raw_df: pd.DataFrame, 
    target_col: str = "Exited", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split raw data into training and validation sets.
    Stratification is applied based on the target column.
    """
    train_df, val_df = train_test_split(
        raw_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=raw_df[target_col]
    )
    return train_df, val_df


def get_feature_columns(train_df: pd.DataFrame, target_col: str = "Exited") -> List[str]:
    """
    Select feature columns for the model.
    Excludes 'id', 'CustomerId', 'Surname', and the target column.
    """
    input_cols = list(train_df.columns)[3:-1]  # skip id, CustomerId, Surname
    return input_cols


def preprocess_data(
    raw_df: pd.DataFrame, 
    scaler_numeric: bool = True
) -> PreprocessedData:
    """
    Perform preprocessing on raw data:
    - Split into training and validation sets
    - Select feature columns
    - Apply one-hot encoding to categorical features
    - Optionally scale numeric features
    
    Returns a PreprocessedData object containing all processed data and fitted transformers.
    """
    train_df, val_df = split_data(raw_df)
    input_cols = get_feature_columns(train_df)
    target_col = "Exited"

    X_train = train_df[input_cols].copy()
    y_train = train_df[target_col].copy()
    X_val = val_df[input_cols].copy()
    y_val = val_df[target_col].copy()

    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_train.select_dtypes("object").columns.tolist()

    # Fit encoder
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(X_train[categorical_cols])

    # Fit scaler (optional)
    scaler = None
    if scaler_numeric:
        scaler = StandardScaler()
        scaler.fit(X_train[numeric_cols])
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    # Transform categorical features
    train_cat = encoder.transform(X_train[categorical_cols])
    val_cat = encoder.transform(X_val[categorical_cols])

    train_cat_df = pd.DataFrame(train_cat, columns=encoder.get_feature_names_out(categorical_cols), index=X_train.index)
    val_cat_df = pd.DataFrame(val_cat, columns=encoder.get_feature_names_out(categorical_cols), index=X_val.index)

    X_train = X_train.drop(columns=categorical_cols).join(train_cat_df)
    X_val = X_val.drop(columns=categorical_cols).join(val_cat_df)

    return PreprocessedData(X_train, y_train, X_val, y_val, input_cols, scaler, encoder)


def preprocess_new_data(
    new_df: pd.DataFrame, 
    preprocessed: PreprocessedData
) -> pd.DataFrame:
    """
    Preprocess new data for prediction using fitted scaler and encoder.
    
    Parameters:
        new_df (pd.DataFrame): New raw data.
        preprocessed (PreprocessedData): Object containing fitted scaler and encoder.
    
    Returns:
        pd.DataFrame: Preprocessed new data ready for prediction.
    """
    new_inputs = new_df[preprocessed.input_cols].copy()

    numeric_cols = new_inputs.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = new_inputs.select_dtypes("object").columns.tolist()

    # Scale numeric features if scaler is available
    if preprocessed.scaler is not None:
        new_inputs[numeric_cols] = preprocessed.scaler.transform(new_inputs[numeric_cols])

    # One-hot transformation
    new_cat = preprocessed.encoder.transform(new_inputs[categorical_cols])
    new_cat_df = pd.DataFrame(new_cat, columns=preprocessed.encoder.get_feature_names_out(categorical_cols), index=new_inputs.index)

    new_inputs = new_inputs.drop(columns=categorical_cols).join(new_cat_df)

    return new_inputs
