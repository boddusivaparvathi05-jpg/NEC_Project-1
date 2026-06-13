import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("customer_dataset.csv")

# FEATURES MUST MATCH YOUR DATASET
X = df[[
    "age",
    "monthly_spend",
    "tenure_months",
    "support_tickets"
]]

y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "churn_model.pkl")

print("Churn model saved successfully")