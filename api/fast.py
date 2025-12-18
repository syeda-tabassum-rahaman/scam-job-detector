"""
Simple fastapi app exposing prediction and explanation endpoints

This module provides a small api to make single job-posting fraud
predictions and return shapley-based explanations
"""

import pandas as pd
from scam_job_detector.ML_logic.model import load_model
from scam_job_detector.ML_logic.data import clean_data
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from scam_job_detector.ML_logic.data import clean_data
from scam_job_detector.ML_logic.preprocessor import test_preprocessor
from scam_job_detector.ML_logic.model import load_model
from scam_job_detector.ML_logic.explainability import explain_xgb
import json
# create api instance
app = FastAPI()

# allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict(
        location: str = None,
        industry: str = None,
        employment_type: str = None,
        has_company_logo: str = None,
        department: str = None,
        description: str = None,
        # job_id: str = None,
        # function_str: str = None,
        # required_experience: str = None,
        # required_education: str = None,
        # has_questions: str = None,
        # salary_range: str = None,
        # title: str = None,
        # company_profile: str = None,
        # requirements: str = None,
        # telecommuting: str = None, 
        # benefits: str = None,  # Predicted JD
    ):
    """
    Make a single prediction and return shapley-based explanations

    Accepts job posting fields as query parameters and returns a fraud
    probability, binary prediction and lists of top shapley features
    """

    # prepare the input DataFrame from query params
    X_new = pd.DataFrame({
        'location': location,
        'industry': industry,
        'employment_type': employment_type,
        'has_company_logo': has_company_logo,
        'department': department,
        'description': description,

        # all of the below listed columns are required to match original dataframe format
        'job_id': None,
        'function': None,
        'required_experience': None,
        'required_education': None,
        'has_questions': None,
        'salary_range': None,
        'title': None,
        'company_profile': None,
        'requirements': None,
        'telecommuting': None,
        'benefits': None
    }, index=[0])

    # clean raw input and transform with fitted preprocessor
    X_new_cleaned = clean_data(X_new)
    X_new_preprocessed = test_preprocessor(X_new_cleaned)

    # load trained model from disk
    model = load_model()

    # generate prediction and probability for positive class
    prediction = model.predict(X_new_preprocessed)[0]
    prediction_proba = model.predict_proba(X_new_preprocessed)[0][1].tolist()

    # # compute shapley explanations for the preprocessed input
    # shap_features_text, shap_text_values, shap_features_binary, shap_values_binary, shap_features_country, shap_values_country = shapley(X_new_preprocessed)


    # return values
    return {
        "fraudulent": int(prediction),
        "prob_fraudulent": float(round(prediction_proba, 4)),
        "column_names": X_new_cleaned.columns.tolist(),
        "column_values": X_new_cleaned.loc[0].tolist()

    }

# @app.get("/explain")
# def explain(column_names: str = None, column_values: str = None):
#     column_names = json.loads(column_names)     # list
#     column_values = json.loads(column_values)   # list

    
#     X_new_cleaned = pd.DataFrame([column_values], columns=column_names)
#     X_new_preprocessed = test_preprocessor(X_new_cleaned)

#     # compute shapley explanations for the preprocessed input
#     shap_features_text, shap_text_values, shap_features_binary, shap_values_binary, shap_features_country, shap_values_country = shapley(X_new_preprocessed)
#     # return values
#     return {
#         'shap_features_text': shap_features_text,
#         'shap_text_values': shap_text_values,
#         'shap_features_binary': shap_features_binary,
#         'shap_values_binary': shap_values_binary,
#         'shap_features_country': shap_features_country,
#         'shap_values_country': shap_values_country
#     }

@app.get("/explain")
def explain(column_names: str = None, column_values: str = None):
    column_names = json.loads(column_names)     # list
    column_values = json.loads(column_values)   # list

    
    X_new_cleaned = pd.DataFrame([column_values], columns=column_names)

    # compute shapley explanations for the preprocessed input
    non_text_contributions, text_contributions_words_fake, text_contributions_contribution_fake, text_contributions_words_real, text_contributions_contribution_real = explain_xgb(X_new_cleaned)
    # return values
    return {
        'non_text_contributions': non_text_contributions,
        'text_contributions_words_fake': text_contributions_words_fake,
        'text_contributions_contribution_fake': text_contributions_contribution_fake,
        'text_contributions_words_real': text_contributions_words_real,
        'text_contributions_contribution_real': text_contributions_contribution_real
    }



#non_text_contributions, text_contributions_words_fake, text_contributions_contribution_fake, text_contributions_words_real, text_contributions_contribution_real

@app.get("/")
def root():
    # simple health endpoint
    return {"greeting": "Hello"}
