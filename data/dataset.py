from datasets import load_dataset
from torch.utils.data import Dataset
from config.config import Config
import logging
logger = logging.getLogger(__name__)

class ChatDataset(Dataset):
    def __init__(self, tokenizer):
        self.data = load_dataset(Config.dataset_name, cache_dir=Config.cache_dir, split="train")
        logger.info(f"Dataset loaded with {len(self.data)} samples.")
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        src = sample["text"]
        tgt = sample["text"]

        src_tokens = self.tokenizer.encode(src, max_length=Config.max_length)["input_ids"].squeeze()
        tgt_tokens = self.tokenizer.encode(tgt, max_length=Config.max_length)["input_ids"].squeeze()

        return {"src": src_tokens, "tgt": tgt_tokens}
