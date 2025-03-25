# 🧠 Chat LLM — Train Your Own Chatbot

This is a modular, object-oriented pipeline for training a custom Transformer-based chatbot using the [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset from Hugging Face. It includes teacher forcing, validation, early stopping, and an interactive chat interface.

---

## 📦 Features

- Custom PyTorch transformer model
- Teacher-forced training
- Validation with early stopping
- Hugging Face tokenizer support (e.g. GPT-2)
- Interactive terminal chatbot
- Clean object-oriented architecture
- Logging with real-time training updates

---

## 🧹 Folder Structure

```text
chat_llm/
├── config/         # Configuration
│   └── config.py   # Hyperparameters and constants
│
├── data/           # Dataset wrapper (Hugging Face)
│   └── dataset.py  # Loads and tokenizes the dataset
│
├── model/          # Transformer + custom layers
│   ├── llm_model.py      # Main transformer model
│   ├── dynamic_tanh.py   # Custom activation layer
│
├── train/          # Training & validation logic
│   ├── trainer.py   # Trainer with early stopping and validation
│   └── validator.py # (optional) Separate validation logic
│
├── utils/          # Tokenizer abstraction
│   └── tokenizer.py # Wrapper around Hugging Face tokenizer
│
├── checkpoints/    # Saved model checkpoints (.pt files)
│
├── chatbot.py      # REPL chatbot interface
├── main.py         # Main entry point: train, validate, and launch chatbot
└── README.md       # Project documentation
```

---

## 🚀 Quick Start

### 1. 🔧 Setup

```bash
pip install torch transformers datasets
```

### 2. 📚 Train the Model

```bash
python chat_llm/main.py
```

- Automatically splits validation data
- Uses teacher forcing
- Early stops if no improvement
- Saves best model to `./checkpoints/model.pt`

### 3. 💬 Chat!

At the end of training, the chatbot launches automatically.

Example:

```
You: Hello!
Bot: Hello! How can I help you today?
```

---

## ⚙️ Configuration

Edit `chat_llm/config/config.py` to adjust:

- Model architecture
- Batch size
- Learning rate
- Max sequence length
- Early stopping patience

---

## 🙏 Acknowledgements

- **Dataset:** This project uses the [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset, provided by Hugging Face and contributors under a suitable open license.
- **Tokenizer:** GPT-2 tokenizer via Hugging Face Transformers.

---

## 🧠 Future Improvements

- Model checkpointing with versioning
- Mixed precision & gradient accumulation
- Web/chat UI with Streamlit or Gradio
- Evaluation metrics (BLEU, perplexity)

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

See the [LICENSE](https://www.gnu.org/licenses/gpl-3.0.en.html) file or visit [gnu.org](https://www.gnu.org/licenses/gpl-3.0.html) for full terms.

This project is for educational and research purposes. Respect the license terms of third-party datasets and models.

