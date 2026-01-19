import os
import pandas as pd 
import polars as pl 
from catboost import CatBoostClassifier, Pool 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import mlflow
import pickle

# Configuration
params = {
    "iterations": 1000,
    "learning_rate": 0.1,
    "depth": 6,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "early_stopping_rounds": 50,
    "verbose": 100
}

def train():
    print("Loading processed data...")
    # Load Data
    df = pl.read_paraquet("data/processed/loan_cleaned.parquet")

    df_pd = df.to_pandas()

    # Features
    X = df_pd.drop(columns=["target"])
    y = df_pd["target"]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical features identified: {cat_features}")

    # Split Data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize MLflow Experiment
    mlflow.set_experiment("LendingClub_Default_Prediction")

    with mlflow.start_run():
        print("Training CatBoost Model...")

        # Initialize Model
        # scale_pos_weight is CRITICAL for imbalanced data (Defaults are rare)
        # It tells the model: "Pay more attention to the minority class (1)"
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = CatBoostClassifier(
            **params,
            cat_features=cat_features,
            scale_pos_weight=pos_weight
        )

        # Train
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True
        )

        # Evaluation
        preds_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, preds_proba)
        print(f"Validation AUC: {auc_score:.4f}")

        # Log Metrics & Params to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("auc", auc_score)
        
        # Save Model Artifacts
        os.makedirs("models", exist_ok=True)
        model_path = "models/catboost_baseline.cbm"
        model.save_model(model_path)
        
        # Log the model file to MLflow
        mlflow.log_artifact(model_path)
        
        print("Training Complete. Model saved.")

if __name__ == "__main__":
    train()