# Text-Summarization-With-HF

---

```markdown
# 📝 Dialogue Summarizer (Fine-tuned T5)

This project is a **Dialogue Summarization Web App** built using a **fine-tuned T5 model** on the [Samsum Dataset](https://huggingface.co/datasets/samsum).  
It takes multi-turn conversations (like chats, interviews, or dialogues) and generates short, meaningful summaries.  

The backend is powered by **FastAPI**, and the frontend is a simple **HTML + JavaScript UI**.  

---

##  Features
- Fine-tuned **T5-small** model for summarization.
- Cleaned and preprocessed **Samsum dataset** (dialogue → summary pairs).
- Summarization inference using **beam search** for better results.
- **FastAPI API** to serve the model.
- Minimal **frontend (HTML/JS)** for easy interaction.

---

## 📂 Project Structure
```
.
├── app.py                 # FastAPI backend
├── index.html             # Frontend UI
├── saved\_summary\_model/   # Fine-tuned model & tokenizer
├── samsum-train.csv       # Training dataset (sampled)
├── samsum-validation.csv  # Validation dataset (sampled)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation

````

---

## ⚙ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dialogue-summarizer.git
cd dialogue-summarizer
````

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` should contain:

```
transformers
torch
fastapi
uvicorn
pydantic
pandas
evaluate
```

---

##  Model Training 

If you want to re-train the model:

1. Load and clean Samsum dataset.
2. Tokenize dialogues and summaries using T5 tokenizer.
3. Fine-tune T5-small with HuggingFace Trainer API.
4. Save the model.

Training script (simplified):

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
# Load tokenizer + model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# TrainingArguments + Trainer setup...
trainer = Trainer(model=model, ...)
trainer.train()

# Save model
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")
```

---

##  Run the App

### 1. Start FastAPI server

```bash
uvicorn app:app --reload
```

API will be live at:
 `http://127.0.0.1:8000`

### 2. Open Frontend

Open `index.html` in your browser.
Type/paste a dialogue and click **Summarize** to get results.

---

##  Example

**Input Dialogue:**

```
Reporter: The IPCC report warns about rising global temperatures.
Expert: Immediate action is needed to cut carbon emissions.
Reporter: What role do individuals play in this?
Expert: Small lifestyle changes collectively make a big difference.
```

**Generated Summary:**

```
The IPCC report warns about climate change. Expert urges action to cut emissions and highlights individual contributions.
```

---

##  Future Improvements

Deploy

---

##  Contributing

Pull requests are welcome! If you’d like to add features, fix bugs, or improve documentation, feel free to open an issue or PR.

---


## Acknowledgements

* [Hugging Face Transformers](https://huggingface.co/transformers/)
* [Samsum Dataset](https://huggingface.co/datasets/samsum)
* [FastAPI](https://fastapi.tiangolo.com/)

---

```

