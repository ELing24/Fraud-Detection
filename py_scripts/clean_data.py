# Phase 1: Loading Files, Cleaning Data, Merging Data into one DataFrame

import pandas as pd  # For structured data
import matplotlib.pyplot as plt  # For creating visualizations 
import json  # For working with JSON

# Load raw data
transactions = pd.read_csv("data/transactions_data.csv")
cards = pd.read_csv("data/cards_data.csv")
users = pd.read_csv("data/users_data.csv")

# Rename columns for merging
transactions.rename(columns={
    "id": "transaction_id",
    "client_id": "user_id",
    "mcc": "mcc_code"
}, inplace=True)

cards.rename(columns={"id": "card_id"}, inplace=True)
users.rename(columns={"id": "user_id"}, inplace=True)

# Load MCC codes JSON and convert to DataFrame
with open("data/mcc_codes.json") as f:
    mcc_data = json.load(f)

mcc_df = pd.DataFrame.from_dict(mcc_data, orient='index', columns=["merchant_category"])
mcc_df.index.name = "mcc_code"
mcc_df.reset_index(inplace=True)
mcc_df["mcc_code"] = mcc_df["mcc_code"].astype(int)

# Load fraud labels from JSON and format
with open("data/train_fraud_labels.json") as f:
    labels_json = json.load(f)

fraud_dict = labels_json["target"]
fraud_df = pd.DataFrame.from_dict(fraud_dict, orient='index', columns=["is_fraud"])
fraud_df.index.name = "transaction_id"
fraud_df.reset_index(inplace=True)
fraud_df["is_fraud"] = fraud_df["is_fraud"].map({"Yes": 1, "No": 0})
fraud_df["transaction_id"] = fraud_df["transaction_id"].astype(int)

# Merge fraud labels into transaction data
df = transactions.merge(fraud_df, on="transaction_id", how="left")
df["is_fraud"] = df["is_fraud"].fillna(0).astype(int)  # Assume unlabeled = not fraud

# Merge card, user, and MCC info
df = df.merge(cards, on="card_id", how="left")
df = df.merge(users, on="user_id", how="left")
df = df.merge(mcc_df, on="mcc_code", how="left")

# Fix numeric types (remove symbols and cast)
for col in ["amount", "credit_limit", "yearly_income", "total_debt", "per_capita_income"]:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

# Drop redundant or duplicate columns
columns_to_drop = [
    "id",                
    "client_id_x",       
    "client_id_y",       
    "mcc",                
    "card_number",       
    "cvv",               
    "address",           
    "merchant_city",     
    "merchant_state",    
    "zip",               
    "birth_year",        
    "birth_month",       
    "retirement_age",    
    "errors"             
]

df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Convert data types for processing
object_cols = df.select_dtypes(include="object").columns
for col in object_cols:
    print(f"\nProcessing column: '{col}'")
    unique_vals = df[col].dropna().unique()

    # Yes/No binary fields
    if pd.Series(unique_vals).str.lower().isin(["yes", "no"]).all():
        print(f"Original unique values: {unique_vals}")
        df[col] = df[col].str.lower().map({"yes": 1, "no": 0})
        print(f"Mapped to binary: {{'yes': 1, 'no': 0}}")
    
    # Categorical fields
    elif df[col].nunique() < 10:
        cat_mapping = {cat: code for code, cat in enumerate(df[col].astype("category").cat.categories)}
        df[col] = df[col].astype("category").cat.codes
        print(f"Original unique values: {list(cat_mapping.keys())}")
        print(f"Mapped to: {cat_mapping}")
    

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["expires"] = pd.to_datetime(df["expires"], errors="coerce")
df["acct_open_date"] = pd.to_datetime(df["acct_open_date"], errors="coerce")

print(df.head())
print(df.info())
print("Data preparation complete.")

# Save as Pickle (preserves datetime, categories, etc.)
df.to_pickle("./outputs/clean_data.pkl")

