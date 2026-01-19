import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import joblib
import os
from catboost import CatBoostClassifier
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# CONFIG
MODEL_PATH = "models/catboost_baseline.cbm"
DATA_PATH = "data/processed/loan_cleaned.parquet"
REPORT_DIR = "reports/figures"

# ---------------------------------------------------------
# ✅ THE WRAPPER CLASS (Must be defined at the top)
# ---------------------------------------------------------
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_  # Required by sklearn check_is_fitted

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def __sklearn_is_fitted__(self):
        return True

# ---------------------------------------------------------

def load_artifacts():
    print("Loading data and model...")
    q = pl.scan_parquet(DATA_PATH)
    df = q.gather_every(10).collect().to_pandas()
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    return df, model

def run_fairness_audit():
    df, model = load_artifacts()
    
    # 1. PREPARE X (Strict Cleaning)
    drop_cols = ["target", "issue_d", "earliest_cr_line"]
    existing_drop = [c for c in drop_cols if c in df.columns] 
    X = df.drop(columns=existing_drop)
    y_true = df["target"]
    
    # Nuclear Fix for Categoricals
    cat_features = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str).replace("nan", "Missing")
    
    sensitive_feature = X["zip_3_digit"].str[0] 
    
    # 2. AUDIT
    print("Predicting for audit...")
    y_pred = model.predict(X)
    
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    
    print("\n--- FAIRNESS AUDIT REPORT ---")
    print(mf.by_group)
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    mf.by_group['selection_rate'].plot(kind='bar', color='skyblue')
    plt.title("Loan Selection Rate by Region")
    plt.savefig(f"{REPORT_DIR}/bias_audit_zip.png")
    
    return df, model, sensitive_feature

def run_mitigation(df, model, sensitive_feature):
    print("\n--- RUNNING BIAS MITIGATION ---")
    
    # 1. Prepare Data
    drop_cols = ["target", "issue_d", "earliest_cr_line"]
    existing_drop = [c for c in drop_cols if c in df.columns] 
    X = df.drop(columns=existing_drop)
    y = df["target"]
    
    # Nuclear Fix for Categoricals
    cat_features = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_features:
        X[col] = X[col].fillna("Missing").astype(str).replace("nan", "Missing")

    # 2. Split
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_feature, test_size=0.3, random_state=42
    )

    # ---------------------------------------------------------
    # 🔴 THE FIX: Filter "Degenerate" Groups
    # We remove groups that don't have BOTH defaults (1) and paid (0)
    # ---------------------------------------------------------
    print("Filtering degenerate groups from training data...")
    valid_groups = []
    for group in A_train.unique():
        # Check target values for this specific group
        y_group = y_train[A_train == group]
        if y_group.nunique() > 1: # Must have both 0 and 1
            valid_groups.append(group)
        else:
            print(f"⚠ Dropping Group '{group}': Only has 1 target class (degenerate).")

    # Apply Filter
    mask = A_train.isin(valid_groups)
    X_train_filtered = X_train[mask]
    y_train_filtered = y_train[mask]
    A_train_filtered = A_train[mask]
    # ---------------------------------------------------------

    # 3. Fit Optimizer (Using Filtered Data)
    compatible_model = CatBoostWrapper(model)

    optimizer = ThresholdOptimizer(
        estimator=compatible_model,
        constraints="demographic_parity", 
        predict_method="predict_proba",
        prefit=True
    )
    
    # Fit on FILTERED data
    optimizer.fit(X_train_filtered, y_train_filtered, sensitive_features=A_train_filtered)
    
    # Evaluate (On Test Data - no need to filter test data usually, but be careful)
    print("Predicting with Fair Model...")
    
    # Only predict for groups the optimizer knows about
    test_mask = A_test.isin(valid_groups)
    y_pred_fair = optimizer.predict(X_test[test_mask], sensitive_features=A_test[test_mask])
    
    mf_fair = MetricFrame(
        metrics={"selection_rate": selection_rate},
        y_true=y_test[test_mask],
        y_pred=y_pred_fair,
        sensitive_features=A_test[test_mask]
    )
    
    print("Results AFTER Mitigation:")
    print(mf_fair.by_group)
    
    joblib.dump(optimizer, "models/fair_model_wrapper.pkl")
    print("Fair model wrapper saved.")

if __name__ == "__main__":
    df, model, sensitive_feature = run_fairness_audit()
    run_mitigation(df, model, sensitive_feature)