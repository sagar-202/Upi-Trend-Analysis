import pandas as pd
import numpy as np

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

print("=== First 5 rows ===")
print(df.head().to_string())

print("\n=== Dataset shape ===")
print(df.shape)

print("\n=== Column names ===")
print(list(df.columns))

print("\n=== Data types ===")
print(df.dtypes)

print("\n=== Missing values ===")
print(df.isnull().sum())

print("\n=== Summary statistics ===")
print(df.describe(include='all').to_string())
