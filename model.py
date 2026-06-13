import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_dataset.csv")

gender_encoder = LabelEncoder()
segment_encoder = LabelEncoder()

df["Gender"] = gender_encoder.fit_transform(df["Gender"])
df["Customer_Segment"] = segment_encoder.fit_transform(
    df["Customer_Segment"]
)

X = df[
    [
        "Age",
        "Gender",
        "Annual_Budget",
        "Browsing_Time",
        "Discount_Sensitivity"
    ]
]

y = df["Customer_Segment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))

joblib.dump(model, "model.pkl")
joblib.dump(segment_encoder, "segment_encoder.pkl")

print("Model Saved Successfully")