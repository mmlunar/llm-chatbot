from transformers import AutoTokenizer

class HFTokenizer:
    def __init__(self, model_path="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode(self, text, max_length=512):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

    def decode(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
