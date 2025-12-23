import pandas as pd
from utils.encryptor import simple_encrypt, redact_text

df = pd.read_csv('IMDB_Dataset.csv')  # Load sample text
df = df.sample(500)  # Keep it small for quick tests

df['text'] = df['text'].apply(lambda x: redact_text(x))
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df[['text', 'label']].to_csv('data/encrypted_dataset.csv', index=False)
