import pandas as pd
import numpy as np

# Load the Adult Income Dataset from UCI repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "ethnicity", "gender",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load dataset
df = pd.read_csv(url, names=columns, sep=",\\s*", engine="python")

# Simplify the dataset for demonstration
df = df[["age", "workclass", "gender", "ethnicity", "income"]].copy()

# Remove missing values
df = df[df != '?'].dropna()

# Display the dataset
print("Original Dataset:")
print(df.head())

# Function to check k-anonymity
def check_k_anonymity(data, quasi_identifiers):
    return data.groupby(quasi_identifiers).size().min()

# Function to check l-diversity
def check_l_diversity(data, quasi_identifiers, sensitive_attribute):
    return data.groupby(quasi_identifiers)[sensitive_attribute].nunique().min()

# Apply k-anonymity
def apply_k_anonymity(data, quasi_identifiers, k):
    data["age"] = (data["age"] // 10) * 10  # Generalizing age to decades
    data["workclass"] = "Public/Private"  # Generalizing workclass
    return data

# Apply l-diversity
def apply_l_diversity(data, quasi_identifiers, sensitive_attribute, l):
    def suppress(group):
        if group[sensitive_attribute].nunique() < l:
            return None
        return group
    return data.groupby(quasi_identifiers).filter(lambda x: suppress(x) is not None)

# Define quasi-identifiers and sensitive attribute
quasi_identifiers = ["age", "workclass", "gender"]
sensitive_attribute = "income"

# Check initial k-anonymity and l-diversity
k = check_k_anonymity(df, quasi_identifiers)
l = check_l_diversity(df, quasi_identifiers, sensitive_attribute)
print(f"Initial k-anonymity: {k}")
print(f"Initial l-diversity: {l}")

# Apply transformations
desired_k = 4
desired_l = 2

df_k_anonymous = apply_k_anonymity(df.copy(), quasi_identifiers, desired_k)
print("\nDataset after applying k-anonymity:")
print(df_k_anonymous.head())

df_l_diverse = apply_l_diversity(df_k_anonymous.copy(), quasi_identifiers, sensitive_attribute, desired_l)
print("\nDataset after applying l-diversity:")
print(df_l_diverse.head())
