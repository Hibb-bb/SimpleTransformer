from bpemb import BPEmb
from torch.utils.data import Dataset, DataLoader

class Tokenizer:
    def __init__(self):
        self.tkr = BPEmb(lang="en", vs=25000)

    def tokenize(self, x):
        assert type(x) is str
        return self.tkr.encode(x)