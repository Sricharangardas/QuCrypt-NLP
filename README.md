# Q-BERT: Encrypted Text Threat Detection using Transformer-Based NLP

## 📌 Overview
Q-BERT is a transformer-based NLP system that analyzes encrypted or redacted text and predicts whether the communication is **Normal** or a **Threat** without decrypting sensitive information.

The project uses a fine-tuned **BERT model** with a custom classification layer to learn contextual patterns from encrypted text.

---

## 🚀 Features
- 🔐 Analyze encrypted or masked text
- 🤖 Transformer-based threat detection
- ⚡ Real-time prediction using Flask API
- 🌐 Interactive frontend using HTML, CSS, JavaScript
- 🧠 Privacy-preserving NLP approach
- 📊 Binary classification:
  - `0 → Normal Communication`
  - `1 → Threat Communication`

---

## 🧩 System Architecture

```text
User Input
    ↓
Tokenization
    ↓
Embeddings (BERT)
    ↓
Transformer Processing (Q-BERT)
    ↓
Classification Layer
    ↓
Prediction Output
```

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Programming Language | Python |
| ML Framework | PyTorch |
| NLP Model | BERT (Transformers) |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Dataset | CSV |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
qbert_project/
│
├── app.py
├── train.py
├── requirements.txt
├── qbert_model.pth
│
├── data/
│   └── encrypted_dataset.csv
│
├── models/
│   └── qbert_model.py
│
├── quantum/
│   └── quantum_embedding.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── script.js
│
└── app/
    └── inference.py
```

---

## 📊 Dataset
The dataset consists of encrypted or redacted sentences labeled as:

| Label | Meaning |
|------|---------|
| 0 | Normal Communication |
| 1 | Threat Communication |

### Example Dataset

```text
"The █████ discussed █████ at ████" → 1
"Routine █████ update completed" → 0
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/Sricharangardas/QuCrypt-NLP.git
cd QuCrypt-NLP
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows
```bash
venv\Scripts\activate
```

#### Linux / Mac
```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

### Example Training Output

```text
Epoch 1 Loss: 0.706
Epoch 2 Loss: 0.509
Epoch 3 Loss: 0.366
```

---

## 🌐 Run the Web Application

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 🧪 Example Predictions

| Input Text | Prediction |
|-----------|-----------|
| The █████ discussed █████ at ████ | Threat |
| Encrypted █████ operation scheduled at ████ | Threat |
| Routine █████ update completed successfully | Normal |
| █████ meeting summary recorded and archived | Normal |

---

## 🔍 How It Works

1. User enters encrypted or redacted text
2. Text is tokenized using BERT tokenizer
3. Tokens are converted into embeddings
4. Transformer model analyzes contextual patterns
5. Classification layer predicts:
   - Normal Communication
   - Threat Communication

---

## 🎯 Real-Life Applications
- Defense encrypted communication analysis
- Privacy-preserving surveillance systems
- Redacted legal document analysis
- Corporate insider threat monitoring
- Secure intelligence systems

---

## ⚠️ Limitations
- Binary classification only
- Depends on training dataset quality
- Uses simulated encrypted text
- Does not categorize threat type

---

## 🚀 Future Scope
- Multi-class threat classification
- Real encrypted communication analysis
- Quantum embedding integration
- Cloud deployment optimization
- Larger encrypted datasets

---

## 👨‍💻 Team Roles

### Member 1
- Model development
- Dataset preparation
- Training and evaluation

### Member 2
- Backend development
- Flask API integration

### Member 3
- Frontend development
- Deployment and UI design

---

## 📜 Conclusion
Q-BERT demonstrates that encrypted or redacted text can be analyzed effectively using transformer-based NLP without revealing sensitive information, enabling privacy-preserving threat detection.

---

## 📧 Contact
- GitHub: https://github.com/Sricharangardas
