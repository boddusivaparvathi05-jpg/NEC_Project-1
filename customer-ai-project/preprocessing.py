import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_prepare_data(path="customer_dataset.csv"):
    """Load the dataset and create basic features for modeling."""
    df = pd.read_csv(path)
    encoder = LabelEncoder()
    df["segment_encoded"] = encoder.fit_transform(df["segment"])
    return df, encoder
