# AI Agent for Text Summarization using BART

This project demonstrates an AI-powered text summarization agent using the pre-trained `facebook/bart-large-cnn` model from Hugging Face. It takes a long input text and generates a concise, meaningful summary using Transformer-based architectures.

---

## Demo

👉 Run the project directly in [Google Colab](https://colab.research.google.com/)  
💬 Paste or type any long paragraph, and it returns a high-quality summary.

---

## Features

- ✔️ Uses BART (Bidirectional and Auto-Regressive Transformers)
- ✔️ Abstractive text summarization
- ✔️ Pre-trained model (`facebook/bart-large-cnn`)
- ✔️ Easy to run in Google Colab
- ✔️ No training required – inference only
- ✔️ Customizable input and output length

---

## Technologies Used

| Tool / Library     | Purpose                          |
|--------------------|----------------------------------|
| Python             | Programming language             |
| Hugging Face       | Transformer models & tokenizer   |
| BART               | Model for summarization          |
| PyTorch            | Backend for model execution      |
| Google Colab       | Execution environment            |

---

## How It Works

1. Input text is tokenized using `BartTokenizer`.
2. The `BartForConditionalGeneration` model processes the tokenized input.
3. Summary is generated using beam search decoding.
4. Output summary is decoded and displayed.

---

## 🧪 Example

**Input:**
> The Indian Space Research Organisation (ISRO) has successfully launched Chandrayaan-3, its third lunar exploration mission...

**Output Summary:**
> ISRO launched Chandrayaan-3, aiming to explore the moon's surface for signs of water and other critical elements.

---

##  Installation

Install the required libraries:

```bash
pip install transformers torch
