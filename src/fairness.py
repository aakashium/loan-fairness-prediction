import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from fairlearn.metrics import MetricFrame, selection_rate, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # To load/save objects

# CONFIG
MODEL_PATH = "models/catboost_baseline.cbm"
DATA_PATH = "data/processed/loan_cleaned.parquet"
REPORT_DIR = "reports/figures"

def load_artifacts():
    print("Loading data and model...")
    # Load Data (Use Polars for speed, convert to Pandas for Fairlearn)
    df = pl.read_parquet(DATA_PATH).sample(fraction=0.1, seed=42).to_pandas()
    
    # Load Model
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    
    return df, model

def run_fairness_audit():
    df, model = load_artifacts()
    
    # Prepare Features
    X = df.drop(columns=["target"])
    y_true = df["target"]
    
    # DEFINING SENSITIVE FEATURES
    # Let's analyze bias based on the first digit of the zip code 
    # (e.g., '9' vs '1' represents West Coast vs East Coast in US)
    sensitive_feature = X["zip_3_digit"].astype(str).str[0] 
    
    # Predict
    y_pred = model.predict(X)
    
    # 1. SETUP METRIC FRAME
    # This acts like a "Pivot Table" for model metrics
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate, # How many loans are approved? (Predicted 0)
        "fnr": false_negative_rate       # How many defaults did we miss?
    }
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    
    # 2. PRINT RESULTS
    print("\n--- FAIRNESS AUDIT REPORT ---")
    print(mf.by_group)
    
    # Check Demographic Parity (Difference in approval rates)
    # Note: In this dataset, target=1 is Default, target=0 is Paid (Approved).
    # So we look at 'selection_rate' carefully.
    max_diff = mf.difference(method='between_groups')['selection_rate']
    print(f"\nMax difference in Selection Rate between regions: {max_diff:.4f}")
    
    # 3. VISUALIZE
    plt.figure(figsize=(10, 6))
    mf.by_group['selection_rate'].plot(kind='bar', color='skyblue')
    plt.title("Loan Selection Rate by Region (Zip Digit)")
    plt.ylabel("Selection Rate (Predicted Default)")
    plt.axhline(y=mf.overall['selection_rate'], color='r', linestyle='--', label='Overall Average')
    plt.legend()
    plt.savefig(f"{REPORT_DIR}/bias_audit_zip.png")
    print(f"Audit plot saved to {REPORT_DIR}/bias_audit_zip.png")

    return df, model, sensitive_feature

def run_mitigation(df, model, sensitive_feature):
    print("\n--- RUNNING BIAS MITIGATION ---")
    
    X = df.drop(columns=["target"])
    y = df["target"]
    
    # Split again for the optimizer
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        X, y, sensitive_feature, test_size=0.3, random_state=42
    )

    # MITIGATION STRATEGY: ThresholdOptimizer
    # Adjusts the decision threshold per group to achieve "Demographic Parity"
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="demographic_parity", 
        predict_method="predict_proba",
        prefit=True
    )
    
    optimizer.fit(X_train, y_train, sensitive_features=A_train)
    
    # Predict using the "Fair" optimizer
    y_pred_fair = optimizer.predict(X_test, sensitive_features=A_test)
    
    # Evaluate Improvement
    mf_fair = MetricFrame(
        metrics={"selection_rate": selection_rate, "accuracy": accuracy_score},
        y_true=y_test,
        y_pred=y_pred_fair,
        sensitive_features=A_test
    )
    
    print("Results AFTER Mitigation:")
    print(mf_fair.by_group)
    
    new_diff = mf_fair.difference(method='between_groups')['selection_rate']
    print(f"\nNew Max difference in Selection Rate: {new_diff:.4f}")
    
    # Save the optimized wrapper
    joblib.dump(optimizer, "models/fair_model_wrapper.pkl")
    print("Fair model wrapper saved.")

if __name__ == "__main__":
    # Ensure report directory exists
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Run Pipeline
    df, model, sensitive_feature = run_fairness_audit()
    run_mitigation(df, model, sensitive_feature)