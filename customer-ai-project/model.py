import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocessing import load_and_prepare_data


def train_model():
    df, _ = load_and_prepare_data()
    X = df[["age", "monthly_spend", "tenure_months", "support_tickets", "segment_encoded"]]
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    print("Saved model.pkl")
    return model


if __name__ == "__main__":
    train_model()
