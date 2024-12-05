import torch
from torch import nn
import math


class EfficientAttention(torch.nn.Module):
    
    def __init__(self, hidden_size:int, dropout=0.1):
        """
        Efficient Attention block that is taken from the paper. 
        Args:
            hidden_size: the embedding dimension
            dropout: the dropout rate
        """
        super().__init__()
        

        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x:torch.Tensor, mask:torch.Tensor):

        # (batch_size, max_len, d_model)
        query = self.query(x) 

        # (batch_size, max_len, d_model) matmul (batch_size, d_model, max_len) --> (batch_size, max_len, max_len)
        scores = torch.matmul(query, x.permute(0, 2, 1)) / math.sqrt(query.size(-1))
        
        
        # fill 0 mask with super small number so it wont affect the softmax weight
        scores = scores.masked_fill(mask == 0, float("-inf")) 
        
        # softmax to put attention weight for all non-pad tokens
        weights = nn.functional.softmax(scores, dim=-1)    
        weights = weights.masked_fill(weights.isnan(), 0)              
        weights = self.dropout(weights)
        
        # (batch_size, max_len, max_len) matmul (batch_size, d_model, max_len) --> (batch_size, d_model, max_len)
        context = torch.matmul(weights, x)
        
        return self.output_linear(context) + x # (batch_size, max_len, d_model) as input
    



class EfficientDifferentialSkipAttention(torch.nn.Module):
    
    def __init__(self, hidden_size:int, kernel_size:int, forward_expansion:float, dropout=0.1):

        super().__init__()
        
        self.d_inner = int(hidden_size*forward_expansion)

        self.dropout = nn.Dropout(dropout)
        self.act = torch.nn.SiLU()

        self.lambda_q1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32).normal_(mean=0,std=0.1))


        self.query = nn.Linear(hidden_size, hidden_size)
        self.att_output = nn.Linear(hidden_size, hidden_size)


        self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=self.d_inner, kernel_size=kernel_size, groups=hidden_size, padding=kernel_size - 1)
        self.contract = nn.Linear(self.d_inner, hidden_size)
        
        

    def forward(self, x:torch.Tensor, mask:torch.Tensor):


        lamb1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lamb2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))

        # (batch_size, max_len, d_model)
        Q = self.query(x) 

        # (batch_size, max_len, d_model) matmul (batch_size, d_model, max_len) --> (batch_size, max_len, max_len)
        scores = torch.matmul(Q, x.permute(0, 2, 1)) / math.sqrt(x.size(-1))        

        # fill 0 mask with super small number so it wont affect the softmax weight
        scores = scores.masked_fill(mask == 0, float("-inf")) 

        # softmax to put attention weight for all non-pad tokens
        weights = nn.functional.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights)

        weights = self.dropout(weights)
        
        # (batch_size, max_len, max_len) matmul (batch_size, d_model, max_len) --> (batch_size, d_model, max_len)
        context = torch.matmul(weights, x)
        
        out1 = self.att_output(context)


        #convolution expansion
        _, L, _ = x.shape
        y = x.transpose(1, 2) # (B, ED, L)
        y = self.conv1d(y)[:, :, :L] # depthwise convolution over time, with a short filter
        y = y.transpose(1, 2) # (B, L, ED)

        #activation
        y = self.act(y)

        #contract
        out2 = self.contract(y)


        out = lamb1 * out1 - lamb2 * out2

        return out # (batch_size, max_len, d_model) as input