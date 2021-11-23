import torch
from torch import nn

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm

from transformers import ViTModel
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.nn.functional as F
from sublayers import MultiHeadAttention, FeedForward, PositionalEncoding
from utils import generate_mask

class ViT(nn.Module):
    def __init__(self, hid_dim=768):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hid_dim=768
        self.proj = nn.Linear(768, 768)

    def forward(self,x):
        features = self.vit(pixel_values=x).last_hidden_state
        return self.proj(features)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x, features, src_mask, trg_mask):
        
        _x = x
        x = x + self.attn_1(x, x, x, trg_mask)
        x = self.dropout(self.ln1(x + _x))

        _x = x
        x = x + self.attn_2(x, features, features, src_mask)
        x = self.dropout(self.ln2(x + _x))

        _x = x
        x = self.ffn(x)
        x = self.dropout(self.ln3(x + _x))

        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, features, src_mask, trg_mask):

        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, features, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        self.encoder = ViT()
        self.decoder = Decoder(vocab_size, d_model, n_layers, heads, dropout)
        self.proj = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, tgt, tgt_mask):
    
        enc_out = self.encoder(img)
        batch_size, patch_num, _ = enc_out.shape

        src_mask = torch.ones(batch_size, patch_num).to(enc_out.device)
        d_output = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        output = self.proj(d_output)
        return self.softmax(output)

# img = torch.rand(4, 3, 224, 224)
# tgt = torch.ones(4, 20, dtype=torch.long)
# mask = tgt
# model = Transformer(vocab_size=20, d_model=768, n_layers=4, heads=4, dropout=0.1)
# pred = model(img, tgt, mask)

