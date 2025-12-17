"""
Shapley value utilities for model explanation

this module computes shap values for a new observation using a kernel
explainer built from a small background sample extracted from the
cleaned dataset
"""

from scam_job_detector.ML_logic.model import load_model, load_preprocessor
import shap
import os
import pandas as pd
import xgboost as xgb

# ====================================================== #
# SHAPLEY VALUES COMPUTATION
# ====================================================== #

def shapley(X_new_preprocessed):
    """
    Compute shap values for a preprocessed observation

    Args:
        X_new_preprocessed: preprocessed feature matrix for the new observation

    Returns:
        tuples of (text features, text shap values, binary features, binary shap values, country features, country shap values)
    """

    # load model and preprocessor from disk
    model = load_model()
    preprocessor = load_preprocessor()

    # load cleaned dataset for building background sample for shap
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")
    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # prediction function required by KernelExplainer (returns positive class probability)
    def model_predict(X_new_preprocessed):
        return model.predict_proba(X_new_preprocessed)[:, 1]

    # build small background dataset by sampling 20 rows from cleaned data
    background_df = (
        df
        .drop(columns=["fraudulent"], errors="ignore")
        .sample(20)
    )

    # transform background with fitted preprocessor and create explainer
    X_background = preprocessor.transform(background_df)
    explainer = shap.KernelExplainer(model_predict, X_background)

    # compute shap values for the provided preprocessed observation
    shap_values = explainer.shap_values(X_new_preprocessed)

    # convert shap output into a tidy dataframe with feature names
    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[0]
    })

    # compute absolute shap value for sorting
    shap_df["abs_value"] = shap_df["shap_value"].abs()

    # identify top features by absolute contribution (for quick inspection)
    top_features = shap_df.sort_values("abs_value", ascending=False).head(20)
    top_features[["feature", "shap_value", "abs_value"]]

    # create masks to separate text, binary and country features
    text_mask = shap_df.feature.str.contains("tfidfvectorizer__")
    binary_mask = shap_df.feature.str.contains("has_company_logo")
    country_mask = shap_df.feature.str.contains("country_")

    text_df = shap_df[text_mask]
    text_df = text_df[text_df.abs_value > 0]
    binary_df = shap_df[binary_mask]
    country_df = shap_df[country_mask]

    # helper functions to extract human-friendly labels from feature names
    def extract_word(feature):
        return feature.split("tfidfvectorizer__")[-1]

    def extract_country(feature):
        return feature.split("country_")[-1]

    def extract_logo(feature):
        return feature.split("has_company_logo_")[-1]

    # apply extraction functions to respective dataframes
    text_df["word"] = text_df["feature"].apply(extract_word)
    binary_df["feature"] = binary_df["feature"].apply(extract_logo)
    country_df["feature"] = country_df["feature"].apply(extract_country)

    binary_mask = shap_df.feature.str.contains("has_company_logo")
    country_mask = shap_df.feature.str.contains("country_")

    shap_features_text = text_df["word"].tolist()
    shap_text_values = text_df["shap_value"].tolist()
    shap_features_binary = binary_df["feature"].tolist()
    shap_values_binary = binary_df["shap_value"].tolist()
    shap_features_country = country_df["feature"].tolist()
    shap_values_country = country_df["shap_value"].tolist()

    return shap_features_text, shap_text_values, shap_features_binary, shap_values_binary, shap_features_country, shap_values_country


# ====================================================== #
# EXPLAIN PREDICTION WITH GRADIENT BOOSTER EXPLAINER
# ====================================================== #

def explain_xgb(X_new):

    # load model and preprocessor from disk
    model = load_model()
    preprocessor = load_preprocessor()

    # preprocess X_new
    X_new_preprocessed = preprocessor.transform(X_new)

    #Get prediction contributions
    contribs = model.get_booster().predict(
        xgb.DMatrix(X_new_preprocessed),
        pred_contribs=True
    )

    # Build contributions dataframe

    feature_names = preprocessor.get_feature_names_out()
    contrib_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contribs[0, :-1]  # exclude bias
    })
    contrib_df["abs_contribution"] = contrib_df.contribution.abs()

    # Create mask to get text, binary, and country features
    xgb_text_mask = pd.DataFrame(contrib_df.feature.str.contains("tfidfvectorizer__"))

    # Apply mask to get dataframes
    xgb_text_df = pd.DataFrame(contrib_df[xgb_text_mask])
    xgb_text_df = pd.DataFrame(xgb_text_df[xgb_text_df.abs_contribution>0])
    xgb_cat_df = pd.DataFrame(contrib_df[~xgb_text_mask])

    #create function to extract non-text contributions
    def extract_non_text_contributions(df):
        explanations = []
        for _, row in df.iterrows():
            if "has_company_logo" in row.feature:
                if "_0" in row.feature:
                    if row.contribution > 0:
                        explanations.append("Missing company logo increases the likelihood that this job posting is fake.")
                    else:
                        explanations.append("Despite the missing company logo, other signals suggest this posting may be legitimate.")
                if "_1" in row.feature:
                    explanations.append("The presence of a company logo increases the credibility of the job posting.")
            elif "country_" in row.feature:
                country = row.feature.split("country_")[-1]
                if row.contribution > 0:
                    explanations.append(f"Job postings originating from {country} are statistically more likely to be fraudulent.")
                else:
                    explanations.append(f"Job postings originating from {country} are generally less associated with fraud in the training data.")
            else:
                explanations.append("No specific explanation available.")
        return explanations

    #Apply function to X_new
    xgb_cat_df["explanation"] =  xgb_cat_df["feature"].apply(extract_non_text_contributions(X_new))

    #Create function to extract word from feature name
    def extract_word(feature):
        return feature.split("tfidfvectorizer__")[-1]

    # Apply function to xgb_text_df
    xgb_text_df["word"] = xgb_text_df["feature"].apply(extract_word)

    # select only top 20 positive contributions
    xgb_text_df_fake = xgb_text_df[xgb_text_df["contribution"]>0].sort_values("contribution",ascending=False).head(20)

    # select only top 20 negative contributions
    xgb_text_df_real = xgb_text_df[xgb_text_df["contribution"]<0].sort_values("contribution",ascending=True).head(20)

    non_text_contributions = xgb_cat_df["explanation"].tolist()
    text_contributions_words_fake = xgb_text_df_fake["word"].tolist()
    text_contributions_contribution_fake = xgb_text_df_fake["contribution"].tolist()
    text_contributions_words_real = xgb_text_df_real["word"].tolist()
    text_contributions_contribution_real = xgb_text_df_real["contribution"].tolist()

    return non_text_contributions, text_contributions_words_fake, text_contributions_contribution_fake, text_contributions_words_real, text_contributions_contribution_real

if __name__ == "__main__":
    explain_xgb()
