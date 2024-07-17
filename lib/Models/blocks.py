import torch
from torch import nn

import math


class SwiGLU(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int):
        r"""
        Simple and original SwiGLU block that is used in the MLP model. Taken from the original paper
        Args:
            input_dim: the input dimension of the block
            hidden_dim: the hidden dimension of the block
        """
        super().__init__()
        self.fcu1 = nn.Linear(input_dim, hidden_dim)
        self.fcu2 = nn.Linear(input_dim, hidden_dim)

        self.fcd1 = nn.Linear(hidden_dim, input_dim)
        self.silu = torch.nn.SiLU()
        
    def forward(self, x):
        x1 = self.fcu1(x)
        x2 = self.fcu2(x)

        y1 = self.silu(x1)

        out = self.fcd1(y1*x2)
        return out
    

    

class EfficientAttention(torch.nn.Module):
    
    def __init__(self, d_model:int, dropout=0.1):
        """
        Efficient Attention block that is taken from the paper. 
        Args:
            d_model: the embedding dimension
            dropout: the dropout rate
        """
        super().__init__()
        

        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor):
        """
        Usually query, key, value are the same tensor.

        Args:
            query: the query tensor of shape (batch_size, max_len, d_model)
            key: the key tensor of shape (batch_size, max_len, d_model)
            value: the value tensor of shape (batch_size, max_len, d_model) 
            mask: the mask tensor of shape (batch_size, max_len, max_len) that contains 0 for padding tokens and 1 for the rest
        """

        # (batch_size, max_len, d_model)
        query = self.query(query) 

        # (batch_size, max_len, d_model) matmul (batch_size, d_model, max_len) --> (batch_size, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / math.sqrt(query.size(-1))
        
        
        # fill 0 mask with super small number so it wont affect the softmax weight
        scores = scores.masked_fill(mask == 0, float("-inf")) 
        
        # softmax to put attention weight for all non-pad tokens
        weights = nn.functional.softmax(scores, dim=-1)           
        weights = self.dropout(weights)
        
        # (batch_size, max_len, max_len) matmul (batch_size, d_model, max_len) --> (batch_size, d_model, max_len)
        context = torch.matmul(weights, value)
        
        return self.output_linear(context) # (batch_size, max_len, d_model) as input
    


class EfficientMultiheadAttention(torch.nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        """
        Args: 
            heads: the number of heads must be a factor of d_model
            d_model: the embedding dimension
            dropout: the dropout rate
        """

        super().__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor):
        """
         Usually query, key, value are the same tensor.

        Args:
            query: the query tensor of shape (batch_size, max_len, d_model)
            key: the key tensor of shape (batch_size, max_len, d_model)
            value: the value tensor of shape (batch_size, max_len, d_model) 
            mask: the mask tensor of shape (batch_size, max_len) that contains 0 for padding tokens and 1 for the rest
        """

        # (batch_size, max_len, d_model)
        query = self.query(query)
 
        
        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        mask = mask.expand_as(scores)# (batch_size, max_len, max_len)

        # fill 0 mask with super small number so it wont affect the softmax weight
        scores = scores.masked_fill(mask == 0, float("-inf")) # (batch_size, h, max_len, max_len)

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = nn.functional.softmax(scores, dim=-1)           
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)
    

class BravBlock(torch.nn.Module):
    def __init__(self, d_model, d_model_expand) -> None:
        """
        Block of new ideation that makes use of few parameters to expand the embedding dimension and 
        then use the expanded dimension to multiply the input tensor.
        >>>>>>MILC Micro Integrated Layer for Classification
        >>>>>>NIMBLE Nano Intelligent Model for Better Layered Extraction
        >>>>>> FLEE
        >>>>>> FLICK / FLEEC
        Args:
            d_model: the embedding dimension
            d_model_expand: the expanded embedding dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_model_expand = d_model_expand

        self.weighter = torch.nn.Parameter(torch.randn(d_model_expand)) #the vector that is used to expand the embedding dimension to a matrix
        self.W_weighter = torch.nn.Linear(d_model, d_model) # the linear layer that is used to scramble the summed matrix (transposed)
        # self.W_weighter = torch.nn.Linear(d_model_expand, d_model_expand)

        self.af = torch.nn.Softsign()

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args: 
            x: the input tensor of shape (batch_size, seq_len, d_model)
            mask: the mask tensor of shape (batch_size, seq_len) that contains 0 for padding tokens and 1 for the rest
        """
        # NOTICE: old stuff        
        # Perform W = x* weighter: (batch_size, seq_len, d_model, d_model_expand)
        #W = (x.unsqueeze(-1) * self.weighter)
        # sum all the matrices along the seq_len axis: (batch_size, d_model, d_model_expand)
        #W_tot = torch.nn.functional.softmax(W.sum(dim=1), dim=1) 


        # Compute (x.unsqueeze(-1) * self.weighter) and its sum along dim=1 in one operation
        # (barch_size, seq_len, d_model) @ (d_model) = (batch_size, d_model, d_model_expand)
        W_tot = torch.einsum('bsd,e->bde', x, self.weighter)

        W_tot = self.af(W_tot)
        #W_tot = torch.nn.functional.softmax(W_tot, dim=1)
        

        W_tot = self.W_weighter(W_tot.transpose(1, 2))
        # W_tot = self.W_weighter(W_tot)

        # use the obtained matrix to multiply all x (W_tot * x)
        # (batch_size, d_model, d_model_expand) @ (batch_size, seq_len, d_model) = (batch_size, seq_len, d_model_expand)
        
        res = torch.matmul(x, W_tot.transpose(1, 2))
        # res = torch.matmul(x, W_tot)

        return res


class MamBravBlock(torch.nn.Module):
    def __init__(self, d_model, state_size) -> None:
        """
        Block of new ideation that tries to build a Mamba block with fewer parameters and smaller activations 

        Args:
            d_model: the embedding dimension
            d_model_expand: the expanded embedding dimension
        """
        super().__init__()
        self.state_size = state_size
        self.fc = torch.nn.Linear(d_model, state_size)
        self.weights = torch.nn.Parameter(torch.randn(state_size))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args: 
            x: the input tensor of shape (batch_size, seq_len, d_model)
            mask: the mask tensor of shape (batch_size, seq_len) that contains 0 for padding tokens and 1 for the rest
        """
        #start by constructing x_bar (batch_size, d_model)
        x_bar = torch.norm(x, dim=1)  #STILL TO DECIDE WHICH IS BEST
        #normalize x_bar (batch_size, d_model)
        x_bar = torch.nn.functional.softmax(x_bar, dim=1) #STILL TO DECIDE WHICH IS BEST
        # x_bar = torch.nn.functional.softplus(x_bar) #STILL TO DECIDE WHICH IS BEST
        # x_bar = torch.nn.functional.elu(x_bar)
        
        #expand x_bar (batch_size, state_size)
        x_bar = self.fc(x_bar)

        #build A_bar (batch_size, state_size, state_size)
        A_bar = torch.einsum('bd,e->bde', x_bar, self.weights)

        #   calculate h'' as x expanded by fully connected  
        h_sec = self.fc(x) #(batch_size, seq_len, state_size)

        h = h_sec[:, 0].unsqueeze(1) # (batch_size, 1, state_size)
        
        #build all complete internal states
        for i in range(1,x.shape[1]):
            # calculate h' as h(t-1) * A_bar
            h_prime = torch.matmul(h[:, -1].unsqueeze(1), A_bar)

            # sum h' and h'' to get h 
            h_prime = torch.nn.functional.tanh(h_prime + h_sec[:, i].unsqueeze(1))
            
            #add new state to h
            h = torch.concat((h, h_prime), dim=1)
            
        # calculate y as the downsample of h
        y = (h - self.fc.bias) @ self.fc.weight

        return y 


class EmbBertAttention(nn.Module):
    def __init__(self, d_model:int, ff_expansion:int, dropout=0.1):
        """
        Efficient Attention block that is taken from the paper and modified to suit Mobile BERT idea 
        Args:
            d_model: the embedding dimension
            dropout: the dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model // ff_expansion)
        
        
    def forward(self, query:torch.Tensor, key:torch.Tensor, value:torch.Tensor, mask:torch.Tensor):
        """
        Usually query, key, value are the same tensor.

        Args:
            query: the query tensor of shape (batch_size, max_len, d_model)
            key: the key tensor of shape (batch_size, max_len, d_model)
            value: the value tensor of shape (batch_size, max_len, d_model) 
            mask: the mask tensor of shape (batch_size, max_len, max_len) that contains 0 for padding tokens and 1 for the rest
        """

        # (batch_size, max_len, d_model)
        query = self.query(query) 

        # (batch_size, max_len, d_model) matmul (batch_size, d_model, max_len) --> (batch_size, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 2, 1)) / math.sqrt(query.size(-1))
        
        
        # fill 0 mask with super small number so it wont affect the softmax weight
        scores = scores.masked_fill(mask == 0, float("-inf")) 
        
        # softmax to put attention weight for all non-pad tokens
        weights = nn.functional.softmax(scores, dim=-1)       
        weights = self.dropout(weights)
        
        # (batch_size, max_len, max_len) matmul (batch_size, d_model, max_len) --> (batch_size, d_model, max_len)
        context = torch.matmul(weights, value)
        
        return self.output_linear(context) # (batch_size, max_len, d_model) as input
    
