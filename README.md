Credit Scoring Model

Objective
Predict an individual's creditworthiness (Good/Bad) using historical financial data.

Dataset
- Source: [UCI German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- 1000 records, 20+ features (income, credit amount, duration, etc.)

Features
- Feature engineering (e.g., debt-income ratio)
- Models: Logistic Regression, Decision Tree, Random Forest
- Metrics: Precision, Recall, F1-score, ROC-AUC

Model Saving

Each trained model is saved in the `models/` folder as a `.pkl` file:
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `random_forest.pkl`

You can reuse them in future apps or APIs using `joblib.load('models/model_name.pkl')`.

Run Locally

```bash
# Setup
pip install -r requirements.txt

# Download dataset
python data/download_data.py

# Train and evaluate model
python scripts/train_model.py

git remote add origin https://github.com/Nidhi181004/credit-scoring-model.git