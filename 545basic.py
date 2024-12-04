import pandas as pd
import numpy as np
from itertools import groupby
import random

# Step 1: Generate the Dataset
def generate_dataset(num_records=100):
    data = {
        "Age": [random.randint(20, 80) for _ in range(num_records)],
        "ZIP Code": [random.randint(12340, 12349) for _ in range(num_records)],
        "Gender": [random.choice(["Male", "Female"]) for _ in range(num_records)],
        "Disease": [random.choice(["Flu", "Diabetes", "Cancer", "Hypertension"]) for _ in range(num_records)],
    }
    return pd.DataFrame(data)

def check_k_anonymity(df, quasi_identifiers):
    grouped = df.groupby(quasi_identifiers).size()
    k = grouped.min()
    return k

def check_l_diversity(df, quasi_identifiers, sensitive_attribute):
    grouped = df.groupby(quasi_identifiers)[sensitive_attribute].apply(lambda x: len(set(x)))
    l = grouped.min()
    return l

def apply_k_anonymity(df, k, quasi_identifiers):
    def generalize_zip(zip_code):
        return str(zip_code)[:3] + "**"

    df["ZIP Code"] = df["ZIP Code"].apply(generalize_zip)
    while check_k_anonymity(df, quasi_identifiers) < k:
        df["Age"] = df["Age"].apply(lambda x: (x // 10) * 10)  # Generalize Age
    return df

def apply_l_diversity(df, l, quasi_identifiers, sensitive_attribute):
    def suppress_rows(group):
        unique_sensitive = group[sensitive_attribute].nunique()
        if unique_sensitive < l:
            group[quasi_identifiers] = None  # Suppress rows not satisfying l-diversity
        return group

    df = df.groupby(quasi_identifiers, group_keys=False).apply(suppress_rows)
    df = df.dropna().reset_index(drop=True)
    return df

# Main Workflow
def main():
    # Generate Dataset
    df = generate_dataset(num_records=100)
    print("Original Dataset:")
    print(df)

    quasi_identifiers = ["Age", "ZIP Code", "Gender"]
    sensitive_attribute = "Disease"

    # Check k and l
    k = check_k_anonymity(df, quasi_identifiers)
    l = check_l_diversity(df, quasi_identifiers, sensitive_attribute)

    print(f"Initial k-anonymity: {k}")
    print(f"Initial l-diversity: {l}")

    # Apply k-anonymity
    desired_k = 2
    df_k_anonymous = apply_k_anonymity(df.copy(), desired_k, quasi_identifiers)

    print(f"Dataset after applying k-anonymity (k={desired_k}):")
    print(df_k_anonymous)

    # Apply l-diversity
    desired_l = 2
    df_l_diverse = apply_l_diversity(df_k_anonymous.copy(), desired_l, quasi_identifiers, sensitive_attribute)

    print(f"Dataset after applying l-diversity (l={desired_l}):")
    print(df_l_diverse)

if __name__ == "__main__":
    main()
