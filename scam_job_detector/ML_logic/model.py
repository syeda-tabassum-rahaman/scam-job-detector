"""
Model training and persistence utilities for the scam-job detector.

This module contains helper functions to run grid searches for baseline
models, train a final model, and load saved models and preprocessors
from disk.
"""

import os
import dill
import pandas as pd
from sklearn.metrics import (
    recall_score, precision_score, balanced_accuracy_score, f1_score,
    average_precision_score, roc_auc_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from scam_job_detector.ML_logic.preprocessor import train_preprocessor, test_preprocessor

def initialize_all_grid_searches(run_logreg=True, run_xgb=True):
    """
    Run grid searches for baseline models and save best estimators

    the function preprocesses the data once, runs optional grid searches
    for logistic regression and xgboost, saves best estimators and selects
    a winner model based on test average precision score
    """

    # load cleaned dataset
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")

    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # train-test split
    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # preprocess once
    X_train_pp = train_preprocessor(X_train)
    X_test_pp = test_preprocessor(X_test)

    # paths for saving models
    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)

    logreg_path = os.path.join(models_folder, "model_logreg.dill")
    xgb_path = os.path.join(models_folder, "model_xgb.dill")
    winner_path = os.path.join(models_folder, "model_winner.dill")

    # store results for winner selection
    model_scores = {}

    # ====================================================== #
    # 1️⃣ logistic regression grid search (if requested)
    # ====================================================== #
    if run_logreg:
        print("\n🔍 Running Logistic Regression Grid Search...")

        # create parameter grid for logistic regression
        param_grid_logreg = {
            "penalty": ["l2"],
            "C": [0.03, 0.1, 0.3, 1, 3, 10],
            "class_weight": [None, "balanced"],
            "solver": ["lbfgs"],
            "max_iter": [2000],
        }

        # create GridSearchCV for logistic regression and run fit
        grid_lr = GridSearchCV(
            LogisticRegression(),
            param_grid_logreg,
            cv=5,
            scoring='average_precision',
            n_jobs=-1
        )
        grid_lr.fit(X_train_pp, y_train)

        # get best estimator and save to disk
        best_lr = grid_lr.best_estimator_

        with open(logreg_path, "wb") as f:
            dill.dump(best_lr, f)

        # results
        print("✅ Grid search for LR completed")

        # inspect best estimator
        print(f"Best score: {grid_lr.best_score_}")
        print(f"Best parameters:, {grid_lr.best_params_}")
        print(f"Best estimator:, {grid_lr.best_estimator_}")

        # evaluate best logistic regression on test set
        y_pred = grid_lr.predict(X_test_pp)
        print(f'''
            Model Performance
            Recall: {recall_score(y_test, y_pred)},
            Precision: {precision_score(y_test, y_pred)},
            Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
            F1 Score: {f1_score(y_test, y_pred)}
            ''')

    else:
        print("\n📂 loading previously saved Logistic Regression model...")
        if os.path.exists(logreg_path):
            with open(logreg_path, "rb") as f:
                best_lr = dill.load(f)
        else:
            best_lr = None

    # evaluate lr if available
    if best_lr is not None:
        y_pred_lr = best_lr.predict(X_test_pp)
        ap_lr = average_precision_score(y_test, y_pred_lr)
        model_scores["logreg"] = (ap_lr, best_lr)
        print(f"🔎 Logistic Regression AP on test: {ap_lr:.4f}")

    # ====================================================== #
    # 2️⃣ XGBOOST GRID SEARCH (if requested)
    # ====================================================== #
    if run_xgb:
        print("\n🔍 Running XGBoost Grid Search...")

        # create parameter grid for xgboost
        param_grid_xgb = {
            "n_estimators": [300, 600],
            "learning_rate": [0.1],
            "max_depth": [10],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.6, 0.8],
            "reg_lambda": [1, 10],
        }
        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1
        )

        # create GridSearchCV for xgboost and run fit
        grid_xgb = GridSearchCV(
            xgb,
            param_grid_xgb,
            cv=5,
            scoring='average_precision',
            n_jobs=2,
            verbose=1
        )
        grid_xgb.fit(X_train_pp, y_train)

        # get best estimator and save to disk
        best_xgb = grid_xgb.best_estimator_

        with open(xgb_path, "wb") as f:
            dill.dump(best_xgb, f)

        print(f"✅ Saved XGBoost model at {xgb_path}")

        # results
        print("✅ Grid search for XGboost completed")

        # inspect best estimator
        print(f"Best score: {grid_xgb.best_score_}")
        print(f"Best parameters:, {grid_xgb.best_params_}")
        print(f"Best estimator:, {grid_xgb.best_estimator_}")

        # evaluate best xgboost on test set
        y_pred = best_xgb.predict(X_test_pp)
        print(f'''
            Model Performance
            Recall: {recall_score(y_test, y_pred)},
            Precision: {precision_score(y_test, y_pred)},
            Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
            F1 Score: {f1_score(y_test, y_pred)}
            ''')


    else:
        print("\n📂 loading previously saved XGBoost model...")
        if os.path.exists(xgb_path):
            with open(xgb_path, "rb") as f:
                best_xgb = dill.load(f)
        else:
            best_xgb = None

    # evaluate xgb if available
    if best_xgb is not None:
        y_pred_xgb = best_xgb.predict(X_test_pp)
        y_pred_xgb_proba = best_xgb.predict_proba(X_test_pp)[:, 1]
        ap_xgb = average_precision_score(y_test, y_pred_xgb_proba)
        model_scores["xgb"] = (ap_xgb, best_xgb)
        print(f"🔎 XGBoost AP on test: {ap_xgb:.4f}")

    # ====================================================== #
    # 3️⃣ CHOOSE THE WINNER MODEL
    # ====================================================== #
    if len(model_scores) == 0:
        print("❌ No models available for comparison. Nothing to save.")
        return None

    winner_name = max(model_scores, key=lambda k: model_scores[k][0])
    winner_score, winner_model = model_scores[winner_name]

    with open(winner_path, "wb") as f:
        dill.dump(winner_model, f)

    print(f"\n🏆 WINNER MODEL: {winner_name}  (AP={winner_score:.4f})")
    print(f"💾 Saved at {winner_path}")

    return None

def final_model():
    # load cleaned dataset
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    clean_data_path = os.path.join(base_path, "raw_data", "data_cleaned.csv")
    df = pd.read_csv(clean_data_path)
    print("✅ Clean data loaded")

    # train-test split
    X = df.drop(columns=["fraudulent"])
    y = df["fraudulent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    # preprocess once
    X_train_pp = train_preprocessor(X_train)
    X_test_pp = test_preprocessor(X_test)

    # paths for saving models
    models_folder = os.path.join(base_path, "models")
    os.makedirs(models_folder, exist_ok=True)

    model_path = os.path.join(models_folder, "final_model.dill")

    # creating weight for class imbalance
    # pos = sum(y_train)
    # neg = len(y_train) - pos
    # spw = neg / pos

    # configure final xgboost model hyperparameters
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        learning_rate=0.05,
        min_child_weight=1,
        reg_lambda=1,
        max_depth=11,
        # scale_pos_weight=spw,
        n_estimators=275
    )
    # train final xgboost model and persist to disk
    xgb.fit(X_train_pp, y_train)
    with open(model_path, "wb") as f:
        dill.dump(xgb, f)

    # evaluate final model on test set
    y_pred = xgb.predict(X_test_pp)
    y_pred_xgb_proba = xgb.predict_proba(X_test_pp)[:, 1]
    print(f'''
        Model Performance
        Recall: {recall_score(y_test, y_pred)},
        Precision: {precision_score(y_test, y_pred)},
        Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred)},
        F1 Score: {f1_score(y_test, y_pred)},
        AUC: {roc_auc_score(y_test, y_pred_xgb_proba)}
        ''' )

def load_model():
    """
    Load the model from the saved final_model file.
    """

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'final_model.dill')

    with open(model_path, "rb") as file:
        model = dill.load(file)
    print("✅ Model loaded")

    return model

def load_preprocessor():
    """
    Load the preprocessor fitted with training data
    """

    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))) , 'models', 'preprocessor.dill')

    with open(model_path, "rb") as file:
        preprocessor = dill.load(file)
    print("✅ Preprocessor loaded")

    return preprocessor

if __name__ == "__main__":
    
    # initialize_all_grid_searches(run_logreg=True, run_xgb=True) # uncomment when you want to use grid search

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_folder = os.path.join(base_path, "models")
    model_path = os.path.join(models_folder, "final_model.dill")

    if os.path.exists(model_path):
        # final_model()
        load_model()
    else:
        final_model()
        load_model()
    
