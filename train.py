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

model = model.cuda()

# print(pred.shape)
opt = torch.optim.Adam(model.parameters(), 0.00005)
cri = nn.CrossEntropyLoss()

trainset = ToyDataset(length=2000)
testset = ToyDataset(length=10)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)
model.train()
for i in range(5):

    p_bar = tqdm(train_loader)
    step, total_loss = 0, 0
    for tgt, img in p_bar:    
        opt.zero_grad()
        tgt, img = tgt.cuda(), img.cuda()
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

    model.eval()
    for tgt, src in loader:
        src, tgt = src.cuda(), tgt.cuda()
        # src_mask = torch.ones(src.size(0), src.size(1)).cuda()
        enc_out = model.encoder(src)
        
        src_mask = torch.ones(enc_out.size(0), enc_out.size(1)).unsqueeze(1).unsqueeze(2).cuda()

        pred_idx = [11]
        for i in range(10):
            tgt_tensor = torch.LongTensor(pred_idx).unsqueeze(0).cuda()
            tgt_mask = make_trg_mask(tgt_tensor)
            # print('tgt input, tgt mask', tgt_tensor.shape, tgt_mask.shape)
            with torch.no_grad():
                d_output, attn_weight = model.decoder(tgt_tensor, enc_out, tgt_mask, src_mask)
                pred_token = d_output[:, 1:].argmax(-1)[:, -1].item()
                pred_idx.append(pred_token)
                # print(pred_token)
                if pred_token == 12:
                    break
        print('gold', tgt.tolist())
        print('pred', pred_idx)
        print('=====================')

inference(model, test_loader)
    
