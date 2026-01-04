import pandas as pd
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Sample encrypted dataset
data = {
    "text": [
        "The █████ met █████ at ████",
        "@#d82Hn!!ds$%",
        "█████ arrived at unknown location",
        "Encrypted $$$$ message detected",
        "The █████ discussed ███████"
    ],
    "label": [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Save dataset
df.to_csv("data/encrypted_dataset.csv", index=False)

print("✅ Dataset created successfully at data/encrypted_dataset.csv")
