import torch
from torch import nn

import lib.Models.modules as modules
import lib.Models.blocks as blocks

class SwiGLU_layer(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, internal_block:nn.Module):
        r"""
        Module with the SwiGLU structure that in parallel of the silu uses another block for computation
        Args:
            input_dim: the input dimension of the block
            hidden_dim: the hidden dimension of the block
            internal_block: the nn.Module that is used in parallel with the silu. The internal block is expected to 
                            have input of shape (batch, seq_len, hidden_dim) and return the same shape 
        """
        super().__init__()
        self.fcu1 = nn.Linear(input_dim, hidden_dim)
        self.fcu2 = nn.Linear(input_dim, hidden_dim)

        self.fcd1 = nn.Linear(hidden_dim, input_dim)
        self.silu = torch.nn.SiLU()
        self.block = internal_block

    def forward(self, x):
        x1 = self.fcu1(x)
        x2 = self.fcu2(x)

        y1 = self.silu(x1)
        y2 = self.block(x2)

        out = self.fcd1(y1*y2)
        return out
    
class Brav_layer(nn.Module):
    def __init__(self, d_model, feed_forward_hidden):
        super().__init__()
        self.norm = modules.RMSNorm(d_model)
        
        self.brav = blocks.BravBlock(d_model, feed_forward_hidden)

        self.up2 = torch.nn.Linear(d_model, feed_forward_hidden)
        self.down1 = torch.nn.Linear(feed_forward_hidden, d_model)
        self.activation = torch.nn.SiLU()

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        y1 = self.brav(embeddings, mask)
        
        y2 = self.up2(embeddings)
        y2 = self.activation(y2)

        y = self.down1(y1 * y2)
        return self.norm(embeddings + y)



class EncoderLayer(nn.Module):
    def __init__(self, attention_module, d_model, ff_expansion, dropout=0.1):
        super(EncoderLayer, self).__init__()
        """
        Layer taken from BERT that concatenates an attention module with skip connection and a feed forward module (SwiGLU) with skip connection
        Args:
            attention_module: the attention module that is used in the layer (must be a nn.Module) and take 
                              (query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor) as parameters
            d_model: the embedding dimension
            ff_expansion: the expansion of the feed forward module
            dropout: the dropout rate
        """
        
        self.layernorm1 = torch.nn.LayerNorm(d_model)
        self.layernorm2 = torch.nn.LayerNorm(d_model)
        self.attention = attention_module
        self.ff = blocks.SwiGLU(d_model, int(ff_expansion*d_model))
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        """
        Args:
            embeddings: the input tensor of shape (batch_size, max_len, d_model)
            mask: the mask tensor of shape (batch_size, max_len) that contains 0 for padding tokens and 1 for the rest
        Returns:
            Tensor of shape (batch_size, max_len, d_model)
        """
        
        attended = self.dropout(self.attention(embeddings, embeddings, embeddings, mask))
        
        # residual layer (skip connection)
        norm_skipped = self.layernorm1(attended + embeddings)

        # bottleneck with skip connection
        feed_forward_out = self.dropout(self.ff(norm_skipped))
        
        encoded = self.layernorm2(feed_forward_out + norm_skipped)
        return encoded
    
class MamBra_layer(nn.Module):
    def __init__(self, d_model, feed_forward_hidden):
        super().__init__()
        # self.norm = modules.RMSNorm(d_model)
        
        self.mambra = blocks.MamBravBlock(d_model, feed_forward_hidden)

        self.up2 = torch.nn.Linear(d_model, feed_forward_hidden)
        self.down1 = torch.nn.Linear(feed_forward_hidden, d_model)
        self.activation = torch.nn.SiLU()

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        y1 = self.mambra(embeddings, mask)
        
        y2 = self.up2(embeddings)
        y2 = self.activation(y2)
        y2 = self.down1(y2)

        return y1 + y2

class EmbBert_layer(nn.Module):
    def __init__(self, d_model, ff_expansion):
        super().__init__()

        #firs stage
        self.contract = nn.Linear(d_model, d_model // ff_expansion)
        self.attention = blocks.EmbBertAttention(d_model, ff_expansion)
        self.norm1 = modules.NoNorm(d_model // ff_expansion)
        

        #second stage
        self.fc = nn.Sequential(
            nn.Linear(d_model // ff_expansion, d_model),
            nn.ReLU(),# SILU in BERT original
            nn.Linear(d_model, d_model // ff_expansion)
        )
        self.norm2 = modules.NoNorm(d_model // ff_expansion)

        
        #final stage
        self.expand = torch.nn.Linear(d_model // ff_expansion, d_model)
        self.norm3 = modules.NoNorm(d_model)
        

    def forward(self, embeddings, mask = None):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        #first stage
        y1 = self.contract(embeddings)
        y2 = self.attention(embeddings, embeddings, embeddings, mask)
        y = self.norm1(y1 + y2)

        #second stage
        y1 = self.fc(y)
        y = self.norm2(y1 + y)

        #final stage
        out = self.expand(y) + embeddings
        
        return self.norm3(out)