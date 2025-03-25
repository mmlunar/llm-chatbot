import os
import torch
import logging
from model.llm_model import LLM_Model
from data.dataset import ChatDataset
from train.trainer import Trainer
from utils.tokenizer import HFTokenizer
from config.config import Config
from chatbot import ChatBot

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

def main():
    tokenizer = HFTokenizer(Config.model_path)
    logger.info("Tokenizer initialized.")

    vocab_size = tokenizer.tokenizer.vocab_size
    model = LLM_Model(vocab_size=vocab_size)
    logger.info(f"Using vocab size: {vocab_size}")

    dataset = ChatDataset(tokenizer)
    logger.info("Starting training...")
    trainer = Trainer(model, dataset, tokenizer)
    trainer.train()

    # Load best model checkpoint
    checkpoint_path = "./checkpoints/model.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=model.embedding.weight.device))
        logger.info(f"✅ Loaded best model from {checkpoint_path}")

    # Launch chatbot
    logger.info("Starting chatbot...")
    bot = ChatBot(model=model, tokenizer=tokenizer)
    bot.chat()

if __name__ == "__main__":
    main()
