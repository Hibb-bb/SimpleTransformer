import torch
import torch.nn as nn
from dataloader import ToyDataset
from vit_cap import Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *

img = torch.rand(4, 3, 224, 224)
tgt = torch.ones(4, 20, dtype=torch.long)
mask = make_trg_mask(tgt)

model = Transformer(13, 768, 8, 12, 0.1)
pred = model(img, tgt, mask)
print(pred.shape)
opt = torch.optim.Adam(model.parameters(), 0.0001)
cri = nn.CrossEntropyLoss()

trainset = ToyDataset()
testset = ToyDataset(length=20)
train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
test_loader = DataLoader(testset, batch_size=4, shuffle=False)

for i in range(5):

    p_bar = tqdm(train_loader)
    step, total_loss = 0, 0
    for tgt, img in p_bar:    
        opt.zero_grad()
        tgt_mask = make_trg_mask(tgt)
        pred = model(img, tgt, tgt_mask)

        token_num = tgt.size(0)* tgt.size(1)
        tgt = tgt.reshape(token_num)
        pred = pred.reshape(token_num, -1)

        loss = cri(pred, tgt.long())
        loss.backward()
        opt.step()
        total_loss += loss.item()
        step += 1

        p_bar.set_description(f'EPOCH {i+1} | {total_loss/step:.4f}')

def inference(model, loader):

    for tgt, src in loader:
        src, tgt = src.to(model.device), tgt.to(model.device)
        src_mask = torch.ones(src.size(0), src.size(1)).to(model.device)
        enc_out = model.encoder(src)
        
        pred_idx = [11]
        for i in range(10):
            tgt_tensor = torch.LongTensor(pred_idx).unsqueeze(0).to(model.device)
            tgt_mask = make_trg_mask(tgt_tensor)
            with torch.no_grad():
                d_output, attn_weight = model.decoder(tgt, enc_out, src_mask, tgt_mask)
                pred_token = d_output.argmax(-1)[:, -1].item()
                pred_idx.append(pred_token)
                if pred_token == 12:
                    break
        print('gold', tgt.tolist())
        print('pred', pred_idx)
    