import pandas as pd
import sys

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

initial_len = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_len - len(df)

df['timestamp'] = pd.to_datetime(df['timestamp'])

df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month

df['transaction_status'] = df['transaction_status'].str.upper()

df['Failure'] = (df['transaction_status'] == 'FAILED').astype(int)

print(f"Duplicates removed: {duplicates_removed}")
print("\n=== Cleaned Dataset Info ===")
df.info(buf=sys.stdout)

print("\n=== First 5 Rows of new columns ===")
print(df[['timestamp', 'hour', 'day', 'month', 'transaction_status', 'Failure']].head().to_string())
