import torch
import torch.nn as nn

from lib.configs import ModelConfig

import lib.Models.structures as structures
import lib.Models.blocks as blocks
import lib.Models.embedders as embedders
import lib.Models.modules as modules

from   lib.Models.mamba import Mamba, MambaConfig


class Mlp_structured(nn.Module):
    
    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the SwiGLU_block
        """
        super().__init__()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.STD_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList([blocks.SwiGLU(model_config.embedding_dimension, model_config.feed_forward_hidden()) for _ in range(model_config.num_layers)])
        self.norms = torch.nn.ModuleList([modules.RMSNorm(model_config.embedding_dimension) for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks with skip connection and normalization
        for encoder, norm in zip(self.encoder_blocks, self.norms):
            x = encoder.forward(x) + x
            x = norm(x)
        return x

class Nano_Mlp_structured(nn.Module):
    
    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the SwiGLU_block
        """
        super().__init__()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList([blocks.SwiGLU(model_config.embedding_dimension, model_config.feed_forward_hidden()) for _ in range(model_config.num_layers)])
        self.norms = torch.nn.ModuleList([modules.RMSNorm(model_config.embedding_dimension) for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks with skip connection and normalization
        for encoder, norm in zip(self.encoder_blocks, self.norms):
            x = encoder.forward(x) + x
            x = norm(x)
        return x



class Brav(nn.Module):

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the Brav_block
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
    
class Gray_BERT_Efficient(nn.Module):
    """
    BERT model with custom gray code embedder
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the BERT_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Gray_nano_embedder(model_config)

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
    


class Mamba_model(nn.Module):
    """
    Mamba model
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the Mamba_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        config = MambaConfig(d_model=model_config.embedding_dimension, 
                             n_layers=model_config.num_layers,
                             expand_factor=model_config.forward_expansion,
                             use_cuda=True )
        self.mamba_layers = Mamba(config)

    def forward(self, x, mask = None):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks
        x = self.mamba_layers(x)
        return x