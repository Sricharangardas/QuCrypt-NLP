from pyexpat import model
import torch
from quantum.quantum_embedding import quantum_embed
def predict(text):
    tokens = quantum_embed(text)
    output = model(tokens['input_ids'], tokens['attention_mask'])
    predicted = torch.argmax(output, dim=1)
    return predicted.item()
