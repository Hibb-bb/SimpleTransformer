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
    def __init__(self):
        
        super().__init__()
        self.postive = torch.tensor([1,2,3,4,5,6,7,8,9,10])
        self.pos_img = torch.triu(torch.rand(3, 224, 224))

        self.negative = torch.tensor([10,9,8,7,6,5,4,3,2,1])
        self.neg_img = torch.tril(torch.rand(3, 224, 224))

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.postive, self.pos_img
        else:
            return self.negative, self.neg_img
    
    def __len__(self):
        return 500