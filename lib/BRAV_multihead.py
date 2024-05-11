import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.MAMBA import RMSNorm

import math

class Embedder(torch.nn.Module):

    def __init__(self, vocab_size, embed_size, reduced_embed_sz = 16, seq_len=128):
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, reduced_embed_sz, padding_idx=0)
        self.expander = nn.Linear(reduced_embed_sz, embed_size)
        self.position = torch.nn.Embedding(seq_len, embed_size)
       
    def forward(self, sequence):
        #build the positions tensor of the tokens
        position = torch.arange(sequence.size(1)).unsqueeze(0).expand_as(sequence).to(sequence.device)

        x = self.expander(self.token(sequence)) + self.position(position)
        return x
    

class Reader_multihead(torch.nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        assert d_model % heads == 0
        self.head_dim = d_model // heads
        #might need to expand and contract the embeddings
        self.weighter = torch.nn.Parameter(torch.randn(self.heads, self.head_dim))
        self.eps = torch.nn.Parameter(torch.randn(self.heads))

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # weight: (d_model, 1)
        res = x.clone()
        x_headed = x.view(x.shape[0], -1, self.heads, self.head_dim)
        res_headed = res.view(x.shape[0], -1, self.heads, self.head_dim).contiguous()

        #res_headed: (batch_size, seq_len, heads, head_dim)
        for i in range(x.size(1)-1):

            W = torch.einsum('bhi,hj->bhij', res_headed[:,i,:,:].clone(), self.weighter) + self.eps.view(-1,1,1) # (batch_size, heads, head_dim, head_dim) 
            
            #res[:,i+1,:] = torch.nn.functional.normalize(torch.einsum('bi,bij->bi', x[:,i+1,:], W), p=2, dim=1).expand_as(res[:,i+1,:])
            res_headed[:,i+1,:,:] += torch.nn.functional.normalize(torch.einsum('bhi,bhij->bhi', x_headed[:,i+1,:,:], W), p=2, dim=1).expand_as(res_headed[:,i+1,:,:])
        

        return res_headed.view(x.shape[0], x.shape[1], -1).contiguous()

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


    
class BravBlock(torch.nn.Module):
    def __init__(self, d_model, n_heads, feed_forward_hidden, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.reader = Reader_multihead(feed_forward_hidden, n_heads)
        self.up1 = torch.nn.Linear(d_model, feed_forward_hidden)
        self.up2 = torch.nn.Linear(d_model, feed_forward_hidden)
        self.down1 = torch.nn.Linear(feed_forward_hidden, d_model)
        self.activation = torch.nn.SiLU()

    def forward(self, embeddings):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)
        x1 = self.norm(embeddings)
        y1 = self.up1(x1)
        y2 = self.up2(x1)
        y1 = self.reader(y1)
        y2 = self.activation(y2)
        x1 = self.down1(y1 * y2)
        return embeddings + x1




class BRAV_multihead(torch.nn.Module):

    def __init__(self, vocab_size, d_model=768, n_heads = 8, n_layers=12, sentence_length = 64, fw_expand = 4,dropout=0.1):


        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = int(d_model * fw_expand)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = Embedder(vocab_size=vocab_size, embed_size=d_model, seq_len= sentence_length)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [BravBlock(d_model, n_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x)
        return x

