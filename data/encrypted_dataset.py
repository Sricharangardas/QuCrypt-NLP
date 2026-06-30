import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import random
from utils.encryptor import simple_encrypt, redact_text

csv_path = 'IMDB_Dataset.csv'

if os.path.exists(csv_path):
    print("Found IMDB_Dataset.csv, generating dataset from it...")
    df = pd.read_csv(csv_path)
    df = df.sample(min(500, len(df)))
    df['text'] = df['text'].apply(lambda x: redact_text(x))
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    df = df[['text', 'label']]
else:
    print("IMDB_Dataset.csv not found. Generating synthetic encrypted dataset...")
    # Generate mock threat and normal communications
    threat_templates = [
        "The {target} discussed {topic} at {location}",
        "Encrypted {topic} operation scheduled at {location}",
        "Planned {topic} operation at {location}",
        "The {target} coordinated {topic} at {location}",
        "Secret {topic} meeting arranged at {location}",
        "Confidential {topic} discussion planned at {location}",
        "The {target} strategy finalized at {location}",
        "Hidden {topic} movement planned at {location}",
        "The {target} task assigned at {location}",
        "Encrypted {topic} communication arranged at {location}",
        "The {target} deployment scheduled at {location}",
        "Secure {topic} briefing planned at {location}",
        "{topic} exchange arranged at {location}",
        "The {target} preparation finalized at {location}",
        "{topic} planning meeting set at {location}",
        "The {target} coordination discussed at {location}",
        "{topic} execution planned at {location}",
        "The {target} confidential update shared at {location}",
        "{topic} strategy meeting held at {location}",
        "The {target} activity confirmed at {location}"
    ]
    
    normal_templates = [
        "Routine {topic} update completed",
        "{topic} meeting summary recorded",
        "Regular {topic} report submitted",
        "{topic} maintenance completed",
        "Routine {topic} communication processed",
        "{topic} report archived successfully",
        "Standard {topic} update shared",
        "{topic} system maintenance logged",
        "Routine {topic} documentation updated",
        "{topic} internal report completed",
        "Regular {topic} schedule confirmed",
        "{topic} service log recorded",
        "Routine {topic} meeting completed",
        "{topic} daily report submitted",
        "Regular {topic} update documented",
        "{topic} status report archived",
        "Routine {topic} check completed",
        "{topic} administrative update recorded",
        "Regular {topic} summary documented",
        "{topic} routine communication logged"
    ]
    
    targets = ["agent", "team", "liaison", "contact", "asset", "commander"]
    topics = ["operation", "strategy", "deployment", "mission", "meeting", "update", "coordination", "tactical"]
    locations = ["location alpha", "sector 7", "hq", "safehouse", "checkpoint", "rendezvous"]
    
    texts = []
    labels = []
    
    for _ in range(150):
        t = random.choice(threat_templates).format(
            target=random.choice(targets),
            topic=random.choice(topics),
            location=random.choice(locations)
        )
        texts.append(redact_text(t))
        labels.append(1)
        
    for _ in range(150):
        t = random.choice(normal_templates).format(
            topic=random.choice(topics)
        )
        texts.append(redact_text(t))
        labels.append(0)
        
    df = pd.DataFrame({"text": texts, "label": labels})
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
df.to_csv('data/encrypted_dataset.csv', index=False)
print("Dataset generated successfully at data/encrypted_dataset.csv")
