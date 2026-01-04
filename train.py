import torch
from torch.utils.data import DataLoader, Dataset
from models.qbert_model import QBertClassifier
from quantum.quantum_embedding import quantum_embed
import pandas as pd

class EncryptedDataset(Dataset):
    def __init__(self, df):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()

    def __getitem__(self, idx):
        tokens = quantum_embed(self.texts[idx])
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.texts)

# Load data
df = pd.read_csv('C:/Users/91934/OneDrive/Documents/GitHub/qbert_project/data/encrypted_dataset.csv')
dataset = EncryptedDataset(df)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = QBertClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
for epoch in range(3):
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['label'])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

import torch
torch.save(model.state_dict(), "qbert_model.pth")

