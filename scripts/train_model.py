# scripts/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load data
X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/target.csv").values.ravel()

# Convert to binary: 1 = good (positive), 0 = bad (negative)
y = np.where(y == 1, 1, 0)


# Rename columns
column_names = [
    'status', 'duration', 'credit_history', 'purpose', 'amount', 'savings',
    'employment', 'installment_rate', 'personal_status_sex', 'other_debtors',
    'present_residence', 'property', 'age', 'other_installment_plans', 'housing',
    'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker'
]
X.columns = column_names

# Feature Engineering
cat_cols = X.select_dtypes(include='object').columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Step 2: Add engineered feature (after encoding)
X['debt_income_ratio'] = X['amount'] / (X['duration'] * X['installment_rate'])
X['debt_income_ratio'] = X['debt_income_ratio'].replace([np.inf, -np.inf], 0)
X['debt_income_ratio'] = X['debt_income_ratio'].fillna(0)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Evaluation
plt.figure(figsize=(10, 6))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n {name} -------------------")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_prob):.2f})")

import joblib
import os

# Create model directory if not exists
os.makedirs("models", exist_ok=True)

# Save each trained model
for name, model in models.items():
    filename = f"models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, filename)
    print(f"Saved model: {filename}")


plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()
