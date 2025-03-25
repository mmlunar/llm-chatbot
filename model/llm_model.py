# llm_model.py

import torch
import torch.nn as nn
import math
from .dynamic_tanh import *

class LLM_Model(nn.Module):
    def __init__(self, vocab_size=128, hidden_dim=512, num_layers=6, num_heads=8, max_len=3161, metadata_len=36):
        super(LLM_Model, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.metadata_len = metadata_len  # Fixed metadata length
        self.max_len = max_len  # Maximum sequence length for generation

        self.embedding = nn.Embedding(vocab_size, hidden_dim).to(device)
        # self.layer_norm = nn.LayerNorm(hidden_dim).to(device)
        self.layer_norm = DynamicTanh(hidden_dim).to(device)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            nhead=num_heads,  # Now explicitly setting the number of attention heads
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        ).to(device)
        self.output_layer = nn.Linear(hidden_dim, vocab_size).to(device)

    def generate_padding_mask(self, tensor):
        """Generate a padding mask (True for padding tokens)"""
        return (tensor == 0) # Assuming 0 is the padding token

    def forward(self, src, tgt):
        """v                      
        - Training mode: Uses `tgt` (Teacher Forcing) starting immediately after `metadata_len`.
        - Inference mode: Generates output after `metadata_len`.
        """
        device = src.device
        src_emb = self.embedding(src.long()) * math.sqrt(self.hidden_dim)
        src_emb = self.layer_norm(src_emb)

        src_padding_mask = self.generate_padding_mask(src).to(device)

        # 🚀 Training Mode (Target starts after metadata)
        tgt_emb = self.embedding(tgt.long()) * math.sqrt(self.hidden_dim)
        tgt_emb = self.layer_norm(tgt_emb)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)

        output = self.transformer(src_emb, tgt_emb, src_key_padding_mask=src_padding_mask, tgt_mask=tgt_mask)

        logits = self.output_layer(output)  # (batch_size, seq_len, vocab_size)
        return logits

