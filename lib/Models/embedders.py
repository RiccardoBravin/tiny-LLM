import torch
from torch import nn
import math

from lib.configs import ModelConfig
from lib.utils import n_ary_gray_code

class STD_embedder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        r"""
        Standard Embedder for transformer models taken from BERT. It is a combination of a token and a positional embedding
        
        Args:
            vocab_size: the size of the vocabulary
            embed_size: the size of the embeddings in output
            max_len: the maximum length of the sequence
            dropout: the dropout rate
        """

        super().__init__()
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(model_config.vocab_size, model_config.embedding_dimension, padding_idx=0)
        self.position = torch.nn.Embedding(model_config.max_length, model_config.embedding_dimension)
       
    def forward(self, tokens):
        #build the positions tensor of the tokens
        positions = torch.arange(tokens.size(1)).unsqueeze(0).expand_as(tokens).to(tokens.device)

        x = self.token(tokens) + self.position(positions)
        return x
    


class Nano_embedder(nn.Module):
    def __init__(self, model_config: ModelConfig):
        r"""
        Embedder for transformer models taken from NanoBERT. It is a combination of a token and a positional embedding where the token embedding
        is done in a lower dimension and then projected to the desired dimension with a linear layer
        
        Args:
            vocab_size: the size of the vocabulary
            embed_size: the size of the embeddings in output
            red_embed_size: the size of the embeddings in the lower dimension
            max_len: the maximum length of the sequence
        """

        super().__init__()
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(model_config.vocab_size, model_config.reduced_embedding_dimension, padding_idx=0)
        self.expander = torch.nn.Linear(model_config.reduced_embedding_dimension, model_config.embedding_dimension, bias=False)
        self.position = torch.nn.Embedding(model_config.max_length, model_config.embedding_dimension)
       
    def forward(self, tokens):
        #build the positions tensor of the tokens
        positions = torch.arange(tokens.size(1)).unsqueeze(0).expand_as(tokens).to(tokens.device)

        x = self.expander(self.token(tokens)) + self.position(positions)
        return x
    