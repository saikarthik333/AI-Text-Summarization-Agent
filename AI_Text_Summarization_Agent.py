!pip install transformers


from transformers import BartTokenizer, BartForConditionalGeneration
import torch


model_name = 'facebook/bart-large-cnn'

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)



def summarize(text, max_input_len=1024, max_output_len=150):
    # Tokenize input
    inputs = tokenizer.batch_encode_plus(
        [text],
        max_length=max_input_len,
        return_tensors='pt',
        truncation=True
    )

    # Generate summary
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_output_len,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    # Decode and return
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


text = input("Enter your text:\n")
summary = summarize(text)
print("\n📋 Summary:\n", summary)
