import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from models.qbert_model import QBertClassifier
from quantum.quantum_embedding import quantum_embed
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

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

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "qbert_model.pth")
    csv_path = os.path.join(base_dir, "data", "encrypted_dataset.csv")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Please run train.py first.")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Error: Dataset file '{csv_path}' not found. Please run create_dataset.py or data/encrypted_dataset.py first.")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model = QBertClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    dataset = EncryptedDataset(df)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    predictions = []
    true_labels = []

    print("Evaluating model...")
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['input_ids'], batch['attention_mask'])
            preds = torch.argmax(outputs, dim=1).tolist()
            predictions.extend(preds)
            true_labels.extend(batch['label'].tolist())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    print("\n--- Evaluation Results ---")
    print(f"Total Evaluated Samples: {len(true_labels)}")
    print(f"Overall Accuracy: {accuracy:.4f}\n")
    print("Classification Report:")
    print(classification_report(true_labels, predictions, target_names=["Normal", "Threat"]))

if __name__ == "__main__":
    main()
