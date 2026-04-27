import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

df = pd.read_csv('model_predictions.csv')
y_true = df['Actual_Failure']
lr_preds = df['LR_Prediction']
rf_preds = df['RF_Prediction']

def eval_model(name, y, preds):
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    cm = confusion_matrix(y, preds)
    
    print(f"=== {name} ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\n")

eval_model("Logistic Regression", y_true, lr_preds)
eval_model("Random Forest Classifier", y_true, rf_preds)
