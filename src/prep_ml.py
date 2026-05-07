import pandas as pd
from sklearn.model_selection import train_test_split

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

# --- Define Features and Target ---
features = ['hour_of_day', 'day_of_week', 'transaction type', 'amount (INR)', 'device_type', 'network_type']
target = 'Failure'

# Derive Target
df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)

# Filter dataset
X_raw = df[features]
y = df[target]

# --- One-Hot Encoding ---
categorical_cols = ['day_of_week', 'transaction type', 'device_type', 'network_type']
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)

# Convert boolean columns to integer (some pandas versions yield bools output for get_dummies)
for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("=== Feature Matrix Shape ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

print("\n=== Target Distribution ===")
print("Overall y distribution:")
print(y.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\ny_train distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\ny_test distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

print("\n=== First 2 Rows of Transformed X_train ===")
print(X_train.head(2).to_string())
