import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('reports/model_predictions.csv')
y_true = df['Actual_Failure']
preds = df['RF_Prediction']

print("=== Deep Validation of Random Forest Model ===")
cm = confusion_matrix(y_true, preds)
print("\n1. Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, preds, zero_division=0))

true_failures = cm[1, 1]
print(f"\n2. How many failures correctly predicted? -> {true_failures}")

total_predicted_failures = cm[0, 1] + cm[1, 1]
total_transactions = len(df)
perc_pred_failures = (total_predicted_failures / total_transactions) * 100
print(f"3. Percentage of failure predictions -> {perc_pred_failures:.2f}%")

total_actual_failures = cm[1, 0] + cm[1, 1]
actual_failure_rate = (total_actual_failures / total_transactions) * 100
print(f"4. Actual Failure Rate vs Predicted:")
print(f"   Actual:    {actual_failure_rate:.2f}%")
print(f"   Predicted: {perc_pred_failures:.2f}%")
