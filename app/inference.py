import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from models.qbert_model import QBertClassifier
from quantum.quantum_embedding import quantum_embed

# Load trained model
model = QBertClassifier()
model.load_state_dict(torch.load("qbert_model.pth", map_location="cpu"))
model.eval()

# Test encrypted input
text = "The █████ discussed █████ at ████"

tokens = quantum_embed(text)

with torch.no_grad():
    output = model(tokens["input_ids"], tokens["attention_mask"])
    prediction = torch.argmax(output, dim=1)

print("Predicted Class:", prediction.item())
