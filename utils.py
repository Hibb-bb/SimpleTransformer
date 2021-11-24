import torch
import numpy as np
from torch.autograd import Variable

def generate_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def make_src_mask(src):
    
    #src = [batch size, src len]
    
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    #src_mask = [batch size, 1, 1, src len]

    return src_mask

def make_trg_mask(trg):
    
    #trg = [batch size, trg len]
    
    trg_pad_mask = (trg != 0).unsqueeze(1).unsqueeze(2)
    
    #trg_pad_mask = [batch size, 1, 1, trg len]
    
    trg_len = trg.shape[1]
    
    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = trg.device)).bool()
    
    #trg_sub_mask = [trg len, trg len]
        
    trg_mask = trg_pad_mask & trg_sub_mask
    
    #trg_mask = [batch size, 1, trg len, trg len]
    
    return trg_mask
