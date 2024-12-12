import pandas as pd
import random

def generate_dataset(num_records=1000):
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
    steps = 0  # Counter for the number of generalization steps
    max_steps = 3  # Allow an additional step for deeper generalization

    while check_k_anonymity(df, quasi_identifiers) < k:
        if steps == 0:
            df["Age"] = df["Age"].apply(lambda x: (x // 10) * 10)  # Generalize Age to 10-year intervals
        elif steps == 1:
            df["Age"] = df["Age"].apply(lambda x: (x // 20) * 20)  # Further generalization to 20-year intervals
        else:
            print(f"Failed to achieve k-anonymity for k={k} after {steps} steps.")
            return None  # Indicating failure
        steps += 1

    return df

def apply_l_diversity(df, l, quasi_identifiers, sensitive_attribute):
    def suppress_rows(group):
        unique_sensitive = group[sensitive_attribute].nunique()
        if unique_sensitive < l:
            return None  # Suppress the group
        return group

    df = df.groupby(quasi_identifiers, group_keys=False).apply(suppress_rows)
    df = df.dropna().reset_index(drop=True)  # Remove suppressed groups
    return df

def main():
    df = generate_dataset(num_records=1000)
    print("Original Dataset:")
    print(df.head())

    quasi_identifiers = ["Age", "ZIP Code", "Gender"]
    sensitive_attribute = "Disease"

    print("\nDistribution of quasi-identifiers:")
    print(df.groupby(["Age", "Gender"]).size())

    k = check_k_anonymity(df, quasi_identifiers)
    l = check_l_diversity(df, quasi_identifiers, sensitive_attribute)

    print(f"\nInitial k-anonymity: {k}")
    print(f"Initial l-diversity: {l}")

    desired_k = 6
    df_k_anonymous = apply_k_anonymity(df.copy(), desired_k, quasi_identifiers)

    if df_k_anonymous is None:
        print(f"Failed to apply k-anonymity for k={desired_k}.")
    else:
        print(f"\nDataset after applying k-anonymity (k={desired_k}):")
        print(df_k_anonymous.head())

        desired_l = 3
        df_l_diverse = apply_l_diversity(df_k_anonymous.copy(), desired_l, quasi_identifiers, sensitive_attribute)

        print(f"\nDataset after applying l-diversity (l={desired_l}):")
        print(df_l_diverse.head())

if __name__ == "__main__":
    main()

