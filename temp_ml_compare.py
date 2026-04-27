import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

file_path = 'dataset/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)
features = ['hour_of_day', 'day_of_week', 'transaction type', 'amount (INR)', 'device_type', 'network_type']
X_raw = df[features]
y = df['Failure']

categorical_cols = ['day_of_week', 'transaction type', 'device_type', 'network_type']
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training RF without Class Weights (Standard)...")
rf_standard = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_standard.fit(X_train, y_train)
preds_standard = rf_standard.predict(X_test)

print("Training RF with class_weight='balanced'...")
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_balanced.fit(X_train, y_train)
preds_balanced = rf_balanced.predict(X_test)

def print_metrics(name, y_true, preds):
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    print(f"=== {name} ===")
    print(f"Recall:    {rec:.4f} (Caught {rec*100:.2f}% of true failures)")
    print(f"Precision: {prec:.4f} (When predicting failure, it was right {prec*100:.2f}% of the time)\n")

print("\n--- RESULTS COMPARISON ---\n")
print_metrics("RF Standard (No Weights)", y_test, preds_standard)
print_metrics("RF Balanced (class_weight='balanced')", y_test, preds_balanced)
