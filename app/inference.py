import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from models.qbert_model import QBertClassifier
from quantum.quantum_embedding import quantum_embed

# Load trained model
model = QBertClassifier()
model.load_state_dict(torch.load("qbert_model.pth", map_location="cpu"))
model.eval()

# Load dataset
df = pd.read_csv("data/encrypted_dataset.csv")

print("\n--- Inference Results ---\n")

for index, row in df.iterrows():
    text = row["text"]

    tokens = quantum_embed(text)

    with torch.no_grad():
        output = model(tokens["input_ids"], tokens["attention_mask"])
        prediction = torch.argmax(output, dim=1).item()

    label_name = "Normal" if prediction == 0 else "Threat"

    print(f"Input: {text}")
    print(f"Predicted Class: {prediction} ({label_name})")
    print("-" * 50)

