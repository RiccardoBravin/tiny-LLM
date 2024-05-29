import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.MAMBA import RMSNorm

import math

class Embedder(torch.nn.Module):

    def __init__(self, vocab_size, embed_size, reduced_embed_size, seq_len):
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, reduced_embed_size, padding_idx=0)
        self.expander = torch.nn.Linear(reduced_embed_size, embed_size)
        self.position = torch.nn.Embedding(seq_len, embed_size)
       
    def forward(self, sequence):
        #build the positions tensor of the tokens
        position = torch.arange(sequence.size(1)).unsqueeze(0).expand_as(sequence).to(sequence.device)

        x = self.expander(self.token(sequence)) + self.position(position)

        return x
    

class Reader(torch.nn.Module):
    def __init__(self, d_model, d_model_expand) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_model_expand = d_model_expand

        self.weighter = torch.nn.Parameter(torch.randn(d_model_expand))
        #self.eps = torch.nn.Parameter(torch.randn(d_model))
        self.W_weighter = torch.nn.Linear(d_model_expand, d_model_expand)

        self.af = torch.nn.Softsign()


    def forward(self, x, x_mask):
        # x: (batch_size, seq_len, d_model)
        # weight: (d_model_expand, 1)
        # res: (batch_size, seq_len, d_model)
    

        # Perform W = (x + epx) * weighter: (batch_size, seq_len, d_model, d_model_expand)
        W = (x).unsqueeze(-1) * self.weighter.unsqueeze(0).unsqueeze(0).unsqueeze(0)


        # sum all the matrices along the seq_len axis: (batch_size, d_model, d_model_expand)

        W_tot = self.af(W.sum(dim=1) / x_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1))
        W_tot = self.W_weighter(W_tot)

        # use the obtained matrix to multiply all x (W_tot * x)
        # (batch_size, d_model, d_model_expand) @ (batch_size, seq_len, d_model) = (batch_size, seq_len, d_model_expand)
        res = torch.einsum('bde,bsd->bse', W_tot, x)

        return res

class FeedForwardSwiGLU(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(d_model, middle_dim)
        self.fc3 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        mid1 = self.activation(self.fc1(x))
        mid2 = self.fc2(x)
        out = self.fc3(mid1 * mid2)
        return out


class oldff(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        mid1 = self.activation(self.fc1(x))
        out = self.fc2(mid1)
        return out


    
class BravBlock(torch.nn.Module):
    def __init__(self, d_model, feed_forward_hidden, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        
        self.reader = Reader(d_model, feed_forward_hidden)

        self.up2 = torch.nn.Linear(d_model, feed_forward_hidden)
        self.down1 = torch.nn.Linear(feed_forward_hidden, d_model)
        self.activation = torch.nn.SiLU()

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)
        x1 = self.norm(embeddings)

        y1 = self.reader(x1, mask)
        y2 = self.up2(x1)
        
        y2 = self.activation(y2)
        x1 = self.down1(y1 * y2)
        return embeddings + x1




class BRAV_2(torch.nn.Module):

    def __init__(self, vocab_size, d_model, red_d_model, n_layers, sentence_length, fw_expand,dropout=0.1):


        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = int(d_model * fw_expand)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = Embedder(vocab_size=vocab_size, embed_size=d_model, reduced_embed_size= red_d_model ,seq_len= sentence_length)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [BravBlock(d_model, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

