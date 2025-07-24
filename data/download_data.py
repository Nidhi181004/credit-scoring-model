from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

dataset = fetch_ucirepo(id=144)  
X = dataset.data.features
y = dataset.data.targets

# Save locally
os.makedirs("data", exist_ok=True)
X.to_csv("data/features.csv", index=False)
y.to_csv("data/target.csv", index=False)
print(" Dataset downloaded and saved.")
