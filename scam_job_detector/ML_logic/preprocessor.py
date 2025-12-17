"""Preprocessing utilities for the scam-job detector.

Builds ColumnTransformer pipelines for categorical, binary and text features,
provides convenience functions to fit/save a preprocessor and to load/apply it
to new data. Saved artifact path: preprocessor.dill.
"""

import os
import dill
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# categorical columns to one-hot encode
categorical_columns = [
    'country',
    'industry',
    'employment_type'
]

# binary columns for simple binary encoding
binary_columns = ['has_company_logo']

# text columns used by the tf-idf transformer
text_columns = ['job_description']

# preprocessor pipeline
def preprocessing_pipeline(text=True, text_only=False) -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    Args:
        text (bool): include text processing when True
        text_only (bool): if True, return a transformer that processes only text columns

    Returns:
        ColumnTransformer: an unfitted transformer combining categorical, binary and optional text transformers
    """
    # pipeline for categorical features: impute missing then one-hot encode
    cat_transformer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='Missing'),
        OneHotEncoder(handle_unknown='ignore')
    )

    # pipeline for binary features: impute most frequent then treat as category
    binary_transformer = make_pipeline(
        SimpleImputer(strategy='most_frequent', fill_value=0),
        OneHotEncoder(handle_unknown='ignore')
    )

    # text transformer built with TfidfVectorizer configured for unigrams and bigrams
    text_transformer = make_column_transformer(
        (TfidfVectorizer(max_df=0.9, min_df=2, ngram_range=(1,2), sublinear_tf=False, max_features=None), "job_description"),
        remainder="drop"
    )

    if text_only:
        # return transformer that only processes text columns
        preprocessor = make_column_transformer((text_transformer, text_columns))
    elif text:
        # full transformer including categorical, binary and text pipelines
        preprocessor = make_column_transformer(
            (cat_transformer, categorical_columns),
            (binary_transformer, binary_columns),
            (text_transformer, text_columns)
        )
    else:
        # transformer excluding text processing
        preprocessor = make_column_transformer(
            (cat_transformer, categorical_columns),
            (binary_transformer, binary_columns),
            remainder='drop'
        )

    return preprocessor

# train preprocessor pipeline
def train_preprocessor(X_train: pd.DataFrame, text=True, text_only = False) -> np.ndarray:
    """Fit and persist the preprocessor on training data.

    Args:
        X_train (pd.DataFrame): feature DataFrame containing expected columns
        text (bool), text_only (bool): forwarded to preprocessing_pipeline

    Returns:
        - transformed training features as numpy array or sparse matrix

    Notes:
        - fitted preprocessor is saved to models/preprocessor.dill using dill
    """

    # build preprocessor and fit on training data
    preprocessor = preprocessing_pipeline(text=text, text_only=text_only)
    X_train_fitted = preprocessor.fit(X_train)
    X_train_preprocessed = X_train_fitted.transform(X_train)

    # persist fitted preprocessor to repository models folder
    preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')
    with open(preprocessor_path, "wb") as file:
        dill.dump(X_train_fitted, file)

    # inform the user where the cleaned data was written
    print(f"✅ Fitted preprocessor saved at: {preprocessor_path}")

    return X_train_preprocessed


# function to load and run the fitted preprocessor on new or test data
def test_preprocessor(X_test: pd.DataFrame) -> np.ndarray:
    """Load saved preprocessor and transform input data.

    Args:
        - X_test (pd.DataFrame): features to transform

    Returns:
        - transformed features as numpy array or sparse matrix
    """

    # load the fitted preprocessor from models folder
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

    with open(model_path, "rb") as file:
        preprocessor = dill.load(file)

    X_test_preprocessed = preprocessor.transform(X_test)
    return X_test_preprocessed

if __name__ == "__main__":
    train_preprocessor()
