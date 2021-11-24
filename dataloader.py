from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader
import torch

class Tokenizer:
    def __init__(self):
        self.tkr = BPEmb(lang="en", vs=25000)

    def tokenize(self, x):
        assert type(x) is str
        return self.tkr.encode(x)


class ToyDataset(Dataset):
    def __init__(self, length=1000):
        
        super().__init__()

        self.l = length
        self.postive = torch.tensor([11,1,2,3,4,5,6,7,8,9,10,12])
        self.pos_img = torch.ones(3, 224, 224)*20

        self.negative = torch.tensor([11,10,9,8,7,6,5,4,3,2,1,12])
        self.neg_img = torch.ones(3, 224, 224)*200

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.postive, self.pos_img
        else:
            return self.negative, self.neg_img
    
    def __len__(self):
        return self.l
