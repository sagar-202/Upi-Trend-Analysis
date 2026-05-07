import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print("Training Logistic Regression (Baseline Model)...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
print("Finished!")

print("Training Random Forest Classifier (Advanced Model)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Finished!")

# Saving predictions securely
results = pd.DataFrame({
    'Actual_Failure': y_test,
    'LR_Prediction': lr_preds,
    'RF_Prediction': rf_preds
})
results.to_csv('reports/model_predictions.csv', index=False)
print(f"Predictions stored successfully in model_predictions.csv. ({len(results)} rows)")

# Saving the Random Forest Model & Columns for Deployment
joblib.dump(rf, 'models/failure_model.pkl')
joblib.dump(list(X.columns), 'models/model_columns.pkl')
print("Model and columns successfully saved for deployment (failure_model.pkl, model_columns.pkl).")
