import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import math
from lib.utils import n_ary_gray_code



class BERTEmbedding(torch.nn.Module):


    def __init__(self, vocab_size, embed_size, reduced_embed_sz = 16, seq_len=128, dropout=0.1):


        super().__init__()
        self.embed_size = embed_size

        log_len = math.ceil(math.log(seq_len, 3))
        #base thre log of the sequence length
        self.pos = torch.Tensor(n_ary_gray_code(log_len)[:seq_len])

        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, reduced_embed_sz, padding_idx=0)
        self.expander = nn.Linear(reduced_embed_sz + log_len, embed_size)
        

        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence):
        #make the position tensor match the sequence dimensions
        position = self.pos.repeat(sequence.size(0), 1, 1).to(sequence.device)

        x = torch.cat((self.token(sequence), position), dim=2)
        x = self.expander(x)
        return self.dropout(x)
    

### attention layers
class EfficientAttention(torch.nn.Module):
    
    def __init__(self, d_model, dropout=0.1):
        super(EfficientAttention, self).__init__()
        

        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        #self.key = nn.Linear(d_model, d_model)
        #self.value = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, max_words, max_words)
        """

        # (batch_size, max_len, d_model)
        query = self.query(query) 

        # (batch_size, max_len, d_model) matmul (batch_size, d_model, max_len) --> (batch_size, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / math.sqrt(query.size(-1))
        
        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        mask = mask.squeeze(1)
        scores = scores.masked_fill(mask == 0, float("-inf")) 

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, max_len, max_len) matmul (batch_size, d_model, max_len) --> (batch_size, d_model, max_len)
        context = torch.matmul(weights, value)
        
        # (batch_size, max_len, d_model)
        return self.output_linear(context)


class FeedForwardSwiGLU(torch.nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForwardSwiGLU, self).__init__()
        
        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.SiLU()

    def forward(self, x):
        mid1 = self.fc1(x)
        mid2 = self.activation(self.fc1(x))
        mul = mid1*mid2
        out = self.fc2(self.dropout(mul))
        return out


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, feed_forward_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = EfficientAttention(d_model, dropout=dropout)
        self.feed_forward = FeedForwardSwiGLU(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer (skip connection)
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck with skip connection
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded
    



class BERT_Eff_gray(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model, reduced_embed_sz, n_layers, sentence_length, fw_expand, dropout=0.1):
        
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = int(d_model * fw_expand)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=sentence_length, reduced_embed_sz = reduced_embed_sz)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

    




