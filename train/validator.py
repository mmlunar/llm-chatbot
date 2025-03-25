import torch
from torch.utils.data import DataLoader
from config.config import Config
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import logging
logger = logging.getLogger(__name__)

class Validator:
    def __init__(self, model, dataset, tokenizer):
        self.model = model
        self.dataset = dataset
        self.tokenizer = tokenizer

    def validate(self, num_samples=5):
        logger.info(f"Running validation on {num_samples} sample(s)...")
        self.model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                sample = self.dataset[i]
                src = sample["src"].unsqueeze(0).to(self.model.embedding.weight.device)
                tgt = sample["tgt"]
                output = self.model(src, src)  # using src as tgt for inference generation
                predicted_ids = output.argmax(dim=-1)[0]
                print(f"> Input: {self.tokenizer.decode(src[0])}")
                print(f"> Predicted: {self.tokenizer.decode(predicted_ids)}")
                print(f"> Ground Truth: {self.tokenizer.decode(tgt)}\n")
