import pandas as pd
import numpy as np

np.random.seed(42)

records = 500

data = []

for i in range(records):

    age = np.random.randint(18, 65)
    gender = np.random.choice(["Male", "Female"])
    budget = np.random.randint(10000, 100000)
    browsing = np.random.randint(5, 60)
    discount = np.random.randint(1, 10)

    if budget > 60000:
        segment = "Luxury"
        product = np.random.choice(
            ["Laptop", "Smartphone", "Gaming Console"]
        )
        purchase = np.random.randint(30000, 80000)

    elif budget > 25000:
        segment = "Medium"
        product = np.random.choice(
            ["Watch", "Shoes", "Headphones"]
        )
        purchase = np.random.randint(10000, 30000)

    else:
        segment = "Budget"
        product = np.random.choice(
            ["T-Shirt", "Groceries", "Household"]
        )
        purchase = np.random.randint(1000, 10000)

    data.append([
        i + 1,
        age,
        gender,
        budget,
        browsing,
        discount,
        product,
        purchase,
        segment
    ])

df = pd.DataFrame(data, columns=[
    "Customer_ID",
    "Age",
    "Gender",
    "Annual_Budget",
    "Browsing_Time",
    "Discount_Sensitivity",
    "Purchased_Product",
    "Purchase_Amount",
    "Customer_Segment"
])

df.to_csv("customer_dataset.csv", index=False)

print("Dataset Generated Successfully")
print(df.head())