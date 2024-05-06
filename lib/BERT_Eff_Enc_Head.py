import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tqdm

import math

class BERTEmbedding(torch.nn.Module):

    def __init__(self, vocab_size, embed_size, reduced_embed_sz = 16, seq_len=128, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, reduced_embed_sz, padding_idx=0)
        self.expander = nn.Linear(reduced_embed_sz, embed_size)
        self.position = torch.nn.Embedding(seq_len, embed_size)
        self.dropout = torch.nn.Dropout(p=dropout)
       
    def forward(self, sequence):
        #build the positions tensor of the tokens
        position = torch.arange(sequence.size(1)).unsqueeze(0).expand_as(sequence).to(sequence.device)

        x = self.expander(self.token(sequence)) + self.position(position)
        return self.dropout(x)
    

### attention layers
class EfficientMultiHeadedAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(EfficientMultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, max_words, max_words)
        """

        # (batch_size, max_len, d_model)
        query = self.query(query)
 
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        scores = scores.masked_fill(mask == 0, float("-inf")) 

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)


class FeedForwardSwiGLU(torch.nn.Module):
    "Implements FFN equation with SiLU."

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
    def __init__(self, d_model, heads, feed_forward_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = EfficientMultiHeadedAttention(heads, d_model)
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
    



class BERT_Eff_Multihead(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model, n_layers, heads ,sentence_length, fw_expand, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = int(d_model * fw_expand)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model, seq_len=sentence_length)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

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

    

class BERT_Eff_multihead_cls(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, sentence_length, fw_expand, n_labels, dropout=0.1):
        super(BERT_Eff_multihead_cls, self).__init__()
        self.model = BERT_Eff_Multihead(vocab_size, d_model, n_layers, heads, sentence_length, fw_expand, dropout)
        self.fc = torch.nn.Linear(d_model, n_labels)

    def forward(self, x):
        x = self.model(x)
        x_mean = x.mean(dim=1)
        out = self.fc(x_mean)
        return out
