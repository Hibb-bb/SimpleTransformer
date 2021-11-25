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
from utils import generate_mask, make_trg_mask

from bpemb import BPEmb

class ViT(nn.Module):
    def __init__(self, hid_dim=768):
        super(ViT, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hid_dim=768
        self.proj = nn.Linear(768, 768)

    def forward(self,x):
        features = self.vit(pixel_values=x).last_hidden_state
        return self.proj(features)

# class DecoderLayer(nn.Module):
#     def __init__(self, d_model, heads, dropout=0.1):
#         super().__init__()

#         self.ln1 = nn.LayerNorm(d_model)
#         self.ln2 = nn.LayerNorm(d_model)
#         self.ln3 = nn.LayerNorm(d_model)
        
#         self.dropout = nn.Dropout(dropout)
        
#         self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
#         self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
#         self.ffn = FeedForward(d_model, dropout=dropout)

#     def forward(self, x, features, src_mask, trg_mask):
        
#         _x = x
#         x = x + self.attn_1(x, x, x, trg_mask)
#         x = self.dropout(self.ln1(x + _x))

#         _x = x
#         x = x + self.attn_2(x, features, features, src_mask)
#         x = self.dropout(self.ln2(x + _x))

#         _x = x
#         x = self.ffn(x)
#         x = self.dropout(self.ln3(x + _x))

#         return x

# class Decoder(nn.Module):
#     def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
#         super().__init__()

#         self.n_layers = n_layers
#         self.embed = nn.Embedding(vocab_size, d_model)
#         self.pe = PositionalEncoding(d_model, dropout=dropout)
#         self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(n_layers)])
#         self.norm = nn.LayerNorm(d_model)

#     def forward(self, trg, features, src_mask, trg_mask):

#         x = self.embed(trg)
#         x = self.pe(x)
#         for i in range(self.n_layers):
#             x = self.layers[i](x, features, src_mask, trg_mask)
#         return self.norm(x)s

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, vocab_size, hid_dim, n_layers, n_heads, dropout, device='cuda', max_length=100):
        super().__init__()
        
        self.device = device
        
        bpemb = BPEmb(lang='en', vs=25000, dim=300, add_pad_emb=True)
        emb_w = torch.FloatTensor(bpemb.emb.vectors)
        extra_tokens = torch.rand(2, 300)
        emb_w = torch.cat((emb_w, extra_tokens), dim=0)
        print('emb weight shape', emb_w.shape) 
        self.tok_embedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb_w, padding_idx=0),
            nn.Linear(300, hid_dim)
        )
        self.tok_embedding = nn.Embedding(vocab_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, dropout, device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]
            
        return output, attention

class Transformer(nn.Module):

    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()

        self.encoder = ViT()
        self.decoder = Decoder(vocab_size, d_model, n_layers, heads, dropout)
        self.proj = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, tgt, tgt_mask):

        enc_out = self.encoder(img)
        batch_size, patch_num, hid = enc_out.shape
        # print('enc out', enc_out.shape)

        src_mask = torch.ones(batch_size, patch_num).unsqueeze(1).unsqueeze(2).to(enc_out.device)
        # print('src mask',src_mask.shape)

        d_output, attn_w = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.softmax(d_output)
