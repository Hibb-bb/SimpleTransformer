import torch.nn as nn

from model.transformer import TransformerBlock
from model.embedding import Embedding

class Transformer(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1, emb_weight=None):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedding(vocab_size=vocab_size, emb_dim=hidden, dropout=dropout, emb_weight=emb_weight)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x

class Classifier(nn.Module):

    def __init__(self, vocab_size, hidden, n_layers, attn_heads, dropout=0.1, class_num=5, emb_weight=None):
        super().__init__()

        self.transformer = Transformer(vocab_size, hidden, n_layers, attn_heads, dropout, emb_weight)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, class_num),
            nn.Softmax(dim=-1)
            )

    def forward(self, x):
        hidden_state = self.transformer(x)
        out = self.mlp(hidden_state[:, 0, :])
        return out

from bpemb.util import sentencepiece_load, load_word2vec_file
from bpemb import BPEmb
import torch
bpemb = BPEmb(lang='en', vs=25000)
bpemb.emb = load_word2vec_file('./model/data/en/en.wiki.bpe.vs25000.d200.w2v.bin')
emb_w = torch.FloatTensor(bpemb.emb.vectors)
print(emb_w.shape)
clf = Classifier(25000, 200, 4, 4, emb_weight=emb_w)