import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier

file_path = 'data/raw/upi_transactions_2024.csv'
df = pd.read_csv(file_path)

df['Failure'] = (df['transaction_status'].str.upper() == 'FAILED').astype(int)
features = ['hour_of_day', 'day_of_week', 'transaction type', 'amount (INR)', 'device_type', 'network_type']
X_raw = df[features]
y = df['Failure']

categorical_cols = ['day_of_week', 'transaction type', 'device_type', 'network_type']
X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
for col in X.select_dtypes(include='bool').columns:
    X[col] = X[col].astype(int)

print("Training Random Forest Classifier on full data to extract importance...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X, y)

importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("\n=== Feature Importances ===")
print(importance_df.to_string(index=False))

# Plot top important features
plt.figure(figsize=(12, 8))
top_n = importance_df.head(10).sort_values(by='Importance', ascending=True)
plt.barh(top_n['Feature'], top_n['Importance'], color='#17a2b8')
plt.xlabel('Importance Score (Gini Importance)')
plt.ylabel('Features')
plt.title('Top 10 Feature Importances (UPI Failures)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

chart_path = os.path.abspath('assets/images/feature_importance.png')
plt.savefig(chart_path)
plt.close()

print(f"\nFeature importance chart saved to: {chart_path}")
