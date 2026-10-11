
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("data/Clean_Dataset.csv")

# Columns to analyze
columns = ["price", "duration", "days_left"]

for column in columns:
    # Convert column to NumPy array and remove missing values
    data = df[column].dropna().to_numpy()

    # Basic statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)

    # Quartiles and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Outlier boundaries
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[
        (data < lower_bound) | (data > upper_bound)
    ]

    # Display results
    print(f"\n--- {column.upper()} ---")
    print(f"Mean: {mean:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Standard deviation: {std:.2f}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower bound: {lower_bound:.2f}")
    print(f"Upper bound: {upper_bound:.2f}")
    print(f"Number of outliers: {len(outliers)}")
    print(f"Outlier percentage: {len(outliers) / len(data) * 100:.2f}%")

    print("First 10 outliers:", outliers[:10])
