import torch
import torch.nn as nn
from dataloader import ToyDataset
from vit_cap import Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

model = Transformer(11, 768, 4, 4, 0.1)
opt = torch.optim.Adam(model.parameters(), 0.0001)
cri = nn.CrossEntropyLoss()

trainset = ToyDataset()
testset = ToyDataset()
train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
test_loader = DataLoader(testset, batch_size=4, shuffle=False)

for i in range(5):

    p_bar = tqdm(train_loader)
    step, total_loss = 0, 0
    for tgt, img in p_bar:    
        opt.zero_grad()
        tgt_mask = torch.ones_like(tgt)
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