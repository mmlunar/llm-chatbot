import torch
from torch.utils.data import DataLoader, random_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from config.config import Config
import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, dataset, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        val_split = int(0.1 * len(dataset))
        train_split = len(dataset) - val_split
        train_data, val_data = random_split(dataset, [train_split, val_split])

        self.train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=Config.batch_size)

        self.optimizer = Adam(model.parameters(), lr=Config.learning_rate)
        self.criterion = CrossEntropyLoss(ignore_index=Config.pad_token_id)
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

    def train(self):
        self.model.train()
        for epoch in range(Config.num_epochs):
            logger.info(f"Epoch {epoch + 1} started.")
            total_loss = 0
            for batch in self.train_loader:
                src = batch["src"].to(self.model.embedding.weight.device)
                tgt = batch["tgt"].to(self.model.embedding.weight.device)

                self.optimizer.zero_grad()
                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

            val_loss = self.validate()
            logger.info(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "./checkpoints/model.pt")
                logger.info("✅ New best model saved.")
            else:
                self.early_stop_counter += 1
                logger.info(f"No improvement. Early stop counter: {self.early_stop_counter}")

            if self.early_stop_counter >= Config.early_stopping_patience:
                logger.info("⏹️ Early stopping triggered.")
                break

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.model.embedding.weight.device)
                tgt = batch["tgt"].to(self.model.embedding.weight.device)

                output = self.model(src, tgt[:, :-1])
                loss = self.criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
                total_val_loss += loss.item()
        self.model.train()
        return total_val_loss / len(self.val_loader)
