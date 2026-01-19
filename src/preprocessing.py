import os
import polars as pl 
from datetime import datetime

def cleaning_data(input_path: str, output_path: str):
    
    # Lazy Load
    q = pl.scan_csv(input_path, ignore_errors = True)

    # Define Leakage & Useless Columns to Drop
    leakage_cols = [
        "recoveries", "collection_recovery_fee", "total_rec_prncp", 
        "total_rec_int", "total_rec_late_fee", "total_pymnt", 
        "total_pymnt_inv", "last_pymnt_d", "last_pymnt_amnt", 
        "next_pymnt_d", "last_credit_pull_d", "debt_settlement_flag",
        "hardship_flag", "pymnt_plan"
    ]

    identifiers = ["id", "member_id", "url", "desc", "policy_code"]

    # Apply Filters
    q = q.drop(leakage_cols + identifiers)

    # Feature Engineering & Type Fixing
    q = q.with_columns([

        pl.col("mths_since_last_delinq").fill_null(-1),
        pl.col("mths_since_last_record").fill_null(-1),
        pl.col("mths_since_last_major_derog").fill_null(-1),

        pl.col("dti").fill_null(18.0), 
        pl.col("revol_util").str.strip_chars("%").cast(pl.Float64).fill_null(50.0),

        pl.col("emp_title").fill_null("Unknown"),
        pl.col("title").fill_null("Unknown"),

        pl.col("int_rate").str.strip_chars("%").cast(pl.Float32), # Interest Rate
        pl.col("emp_length").str.extract(r"(\d+)", 1).cast(pl.Int32).fill_null(0), # Employment Length
        pl.col("zip_code").str.slice(0, 3).alias("zip_3_digit"), # Zip Code

    ])

    q = q.with_columns([
        pl.col("issue_d").str.to_date("%b-%Y", strict=False),
        pl.col("earliest_cr_line").str.to_date("%b-%Y", strict=False)
    ])

    q = q.with_columns([
        # Credit History Length (Crucial Feature)
        # (Loan Issue Date - Earliest Credit Line) in years
        ((pl.col("issue_d") - pl.col("earliest_cr_line")).dt.total_days() / 365.25)
        .alias("credit_hist_years"),

        # Flag: Has the borrower ever been delinquent? (Interaction)
        pl.when(pl.col("mths_since_last_delinq") == -1)
          .then(0)
          .otherwise(1)
          .alias("has_delinq_history"),
        
        # Income to Loan Ratio (Affordability)
        (pl.col("annual_inc") / (pl.col("loan_amnt") + 1)).alias("income_to_loan_ratio"),
        
        # Target Engineering (1 = Default, 0 = Paid)
        pl.when(pl.col("loan_status").is_in(["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"]))
          .then(1)
          .when(pl.col("loan_status").is_in(["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]))
          .then(0)
          .otherwise(None)
          .alias("target")
    ])

    # Drop rows where target is undefined 
    q = q.drop_nulls(subset=["target"])
    
    # Execute and Save
    print("Processing and saving...")
    q = q.fetch(200_000)
    df = q.collect(streaming=True) # This triggers the computation
    df.sink_parquet(output_path) 
    print(f"Cleaned data saved to {output_path} with shape: {df.shape}")

if __name__ == "__main__":
    # Example usage
    clean_lending_club_data("data/raw/loan.csv", "data/processed/loan_cleaned.parquet")
