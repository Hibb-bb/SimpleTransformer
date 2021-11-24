import torch
import torch.nn as nn
from dataloader import ToyDataset
from vit_cap import Transformer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import *

from dataset import get_loader
import argparse
from torchvision import transforms

from bpemb import BPEmb

def main(args):
    
    model = Transformer(25002, 768, 8, 12, 0.1)
    opt = torch.optim.Adam(model.parameters(), 0.00005)
    cri = nn.CrossEntropyLoss()

    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    data_loader = get_loader(args.image_dir, args.caption_path, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    for i in range(args.num_epochs):

        total_step, total_loss = 0, 0

        for img, tgt in data_loader:
            
            opt.zero_grad()
            img, tgt = img.cuda(), tgt.cuda()
            pred = model(img, tgt, make_trg_mask(tgt))
            pred = pred.reshape(-1, 25002)
            label = tgt.reshape(-1)
            loss = cri(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_step += 1

            if total_step % 100 == 0:
                print(f'EPOCH {i+1} STEP {total_step} Loss {(total_loss/total_step):.4f}')

            sample_loader = get_loader(args.image_dir, args.caption_path, 
                                    transform, 1,
                                    shuffle=True, num_workers=args.num_workers) 

            inference(model, sample_loader[:15])

def inference(model, loader):

    bpemb_en = BPEmb(lang="en", vs=25000)
    model.eval()
    for src, tgt in loader:
        
        src, tgt = src.cuda(), tgt.cuda()
        # src_mask = torch.ones(src.size(0), src.size(1)).cuda()
        enc_out = model.encoder(src)
        
        src_mask = torch.ones(enc_out.size(0), enc_out.size(1)).unsqueeze(1).unsqueeze(2).cuda()

        # src, tgt = src.cuda(), tgt.cuda()
         # .cuda()
        # enc_out = model.encoder(src)
        # src_mask = torch.ones(enc_out.size(0), enc_out.size(1)).unsqueeze(1).unsqueeze(2)
        # print('enc out', enc_out.shape)
        
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
        print('gold', bpemb_en.decode_ids(tgt.tolist()))
        print('pred', bpemb_en.decode_ids(pred_idx))
        print('=====================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='data/resized2014', help='directory for resized images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
