import os
import polars as pl 
from datetime import datetime

def cleaning_data(input_path: str, output_path: str):

    q = pl.scan_csv(
        input_path, 
        ignore_errors=True,
        schema_overrides={
            "revol_util": pl.String,
            "int_rate": pl.String,
            "emp_length": pl.String 
        })

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

    # Processed data
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Execute and Save
    print("Processing and saving...")
    q.sink_parquet(output_path) 
    saved_q = pl.scan_parquet(output_path)
    
    # Count rows efficiently without loading the whole file
    row_count = saved_q.select(pl.len()).collect().item()
    col_count = len(saved_q.columns)
    
    print(f"Cleaned data saved to {output_path}")
    print(f"Final Shape: ({row_count} rows, {col_count} cols)")

if __name__ == "__main__":
    # Example usage
    cleaning_data("data/Raw/accepted_2007_to_2018Q4.csv", "data/processed/loan_cleaned.parquet")
