import torch
import torch.nn as nn

from lib.configs import ModelConfig

import lib.Models.structures as structures
import lib.Models.blocks as blocks
import lib.Models.embedders as embedders



class Brav(nn.Module):

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the Brav_block
        Args:
            vocab_size: the size of the vocabulary
            d_model: the embedding dimension
            red_d_model: the reduced embedding dimension for NanoBERT embedder
            sentence_length: the length of the sentence
            fw_expand: the expansion factor of the feed forward network
            n_layers: the number of layers
            dropout: the dropout rate
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.layers = torch.nn.ModuleList(
            [structures.Brav_layer(model_config.embedding_dimension, model_config.feed_forward_hidden()) for _ in range(model_config.num_layers)]
        )

    def forward(self, x, mask):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks
        for encoder in self.layers:
            x = encoder.forward(x, mask)
        return x



class Bert_efficient(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the BERT_block
        Args:
            vocab_size: the size of the vocabulary
            d_model: the embedding dimension
            n_layers: the number of layers
            sentence_length: the length of the sentence
            fw_expand: the expansion factor of the feed forward network
            dropout: the dropout rate
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.STD_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [structures.EncoderLayer(
                        blocks.EfficientAttention(model_config.embedding_dimension, dropout), 
                        model_config.embedding_dimension, 
                        model_config.forward_expansion, 
                        dropout)
                                    for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    
    

class Nano_Bert_Efficient(nn.Module):
    """
    BERT model with NanoBERT embedder
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the BERT_block
        Args:
            vocab_size: the size of the vocabulary
            d_model: the embedding dimension
            n_layers: the number of layers
            sentence_length: the length of the sentence
            fw_expand: the expansion factor of the feed forward network
            dropout: the dropout rate
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [structures.EncoderLayer(
                            blocks.EfficientAttention(model_config.embedding_dimension, dropout), 
                            model_config.embedding_dimension, 
                            model_config.forward_expansion, 
                            dropout) 
                                    for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x