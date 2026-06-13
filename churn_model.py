import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("customer_dataset.csv")

encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Create churn column if missing
if "Churn" not in df.columns:
    import numpy as np
    df["Churn"] = np.random.choice([0,1], size=len(df), p=[0.8,0.2])

X = df[
[
    "Age",
    "Annual_Budget",
    "Browsing_Time",
    "Discount_Sensitivity",
    "Purchase_Amount"
]
]

y = df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Churn Accuracy:", accuracy_score(y_test,pred))

joblib.dump(model,"churn_model.pkl")

print("Churn Model Saved Successfully")