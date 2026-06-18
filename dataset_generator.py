import pandas as pd
import numpy as np

np.random.seed(42)

n = 500

df = pd.DataFrame({
    "age": np.random.randint(18, 65, n),
    "monthly_spend": np.random.randint(5000, 50000, n),
    "tenure_months": np.random.randint(1, 60, n),
    "support_tickets": np.random.randint(0, 10, n),
    "segment": np.random.choice(
        ["Budget", "Medium", "Premium"],
        n
    ),
    "churn": np.random.choice(
        [0, 1],
        n,
        p=[0.8, 0.2]
    )
})

df.to_csv("customer_dataset.csv", index=False)

print("Dataset Generated Successfully")
print(df.head())