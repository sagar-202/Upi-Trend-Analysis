import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score

file_path = 'dataset/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

# Target
df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)

# NEW Engineered Features
df['Is_Peak_Failure_Hour'] = (df['hour_of_day'] == 6).astype(int)
df['Is_Weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

features = ['hour_of_day', 'day_of_week', 'transaction type', 'amount (INR)', 'device_type', 'network_type', 'Is_Peak_Failure_Hour', 'Is_Weekend']
X_raw = df[features]
y = df['Failure']

categorical_cols = ['day_of_week', 'transaction type', 'device_type', 'network_type']
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training RF with New Engineered Features...")
rf_engineered = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_engineered.fit(X_train, y_train)
preds_engineered = rf_engineered.predict(X_test)

new_rec = recall_score(y_test, preds_engineered, zero_division=0)
new_prec = precision_score(y_test, preds_engineered, zero_division=0)

print("\n--- RESULTS WITH ENGINEERED FEATURES ---")
print(f"Recall:    {new_rec:.4f} (Caught {new_rec*100:.2f}% of true failures)")
print(f"Precision: {new_prec:.4f} (When predicting failure, it was right {new_prec*100:.2f}% of the time)\n")

print("--- Comparisons against Previous RF (Balanced) ---")
print(f"Previous Recall was ~0.0291 (2.91%).")
print(f"New Recall is {new_rec:.4f} ({new_rec*100:.2f}%).")
print(f"Absolute Improvement: +{(new_rec - 0.0291)*100:.2f}%")
