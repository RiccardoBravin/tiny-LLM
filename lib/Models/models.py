import torch
import torch.nn as nn

from lib.configs import ModelConfig

import lib.Models.structures as structures
import lib.Models.blocks as blocks
import lib.Models.embedders as embedders
import lib.Models.modules as modules

from   lib.Models.mamba import Mamba, MambaConfig


class BERT_original(nn.Module):
    
    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the SwiGLU_block
        """
        super().__init__()
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.STD_embedder(model_config)

        # multi-layers transformer blocks, deep network
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_config.embedding_dimension, 
                                                   nhead=model_config.number_of_heads,
                                                   dim_feedforward=model_config.feed_forward_hidden(),
                                                   batch_first=True,
                                                   dropout=dropout) 
 
        self.bert_layers = torch.nn.TransformerEncoder(encoder_layer, num_layers=model_config.num_layers, enable_nested_tensor=False)
    
    
    def forward(self, x, mask):
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)
        
        # applying the transformer layers
        x = self.bert_layers(x)

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
    BERT model with NanoBERT embedder and efficient attention
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
        mask = modules.make_score_mask(mask)
        
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x

class Nano_Bert_Efficient_mh(nn.Module):
    """
    BERT model with NanoBERT embedder and efficient multihead attention 
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
                            blocks.EfficientMultiheadAttention(model_config.number_of_heads, model_config.embedding_dimension, dropout), 
                            model_config.embedding_dimension, 
                            model_config.forward_expansion, 
                            dropout) 
                                    for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = modules.make_score_mask(mask)
        
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

        self.dropout = nn.Dropout(dropout)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        config = MambaConfig(d_model=model_config.embedding_dimension, 
                             n_layers=model_config.num_layers,
                             expand_factor=model_config.forward_expansion,
                             d_state=model_config.d_state,
                             d_conv=4,
                             use_cuda=True )
        self.mamba_layers = Mamba(config)

    def forward(self, x, mask = None):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)
        
        x = self.dropout(x)

        # running over multiple transformer blocks
        x = self.mamba_layers(x)
        return x



class MamBra_model(nn.Module):
    """
    MamBra model
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the MamBra_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        # multi-layers transformer blocks, deep network
        self.mambra_layers = torch.nn.ModuleList(
            [structures.MamBra_layer(model_config.embedding_dimension, model_config.feed_forward_hidden()) for _ in range(model_config.num_layers)]
        )

    def forward(self, x, mask = None):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # running over multiple transformer blocks
        for encoder in self.mambra_layers:
            x = encoder.forward(x, mask) # + x
        return x

class Embedder_model(nn.Module):
    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the MamBra_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)


    def forward(self, x, mask = None):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        return x


class Embedder_conv_model(nn.Module):
    
    def __init__(self, model_config: ModelConfig, kernel_sz = 16, dropout=0.1):
        """
        Embedder and multilayer model using the MamBra_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder(model_config)

        self.d_conv = kernel_sz

        self.conv1d = nn.Conv1d(in_channels=model_config.embedding_dimension, out_channels=model_config.embedding_dimension, 
                              kernel_size=self.d_conv,
                              groups=model_config.embedding_dimension,
                              padding=self.d_conv - 1)
        

    def forward(self, x, mask = None):
        
        l = x.shape[1]

        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)
        
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:,:,:l]
        x = x.transpose(1, 2)

        return x

class Embbert(nn.Module):
    """
    BERT inspired model that uses NanoBERT embedder, Efficient attention and Bottleneck and Inverted Bottleneck of MobileBERT
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
        self.embbert_layers = torch.nn.ModuleList(
            [structures.EmbBert_layer(model_config.embedding_dimension, model_config.feed_forward_hidden()) for _ in range(model_config.num_layers)]
        )

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = mask.unsqueeze(1).repeat(1, x.shape[1], 1)

        # running over multiple transformer blocks
        for layer in self.embbert_layers:
            x = layer.forward(x, mask)
        return x
    

class Nano_Bert_Efficient_mh_augm(nn.Module):
    """
    BERT model with NanoBERT embedder and efficient multihead attention 
    """
    
    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the BERT_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.Nano_embedder_augmentation(model_config)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [structures.EncoderLayer(
                            blocks.EfficientMultiheadAttention(model_config.number_of_heads, model_config.embedding_dimension, dropout), 
                            model_config.embedding_dimension, 
                            model_config.forward_expansion, 
                            dropout) 
                                    for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = modules.make_score_mask(mask)
        
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x



class Mamba_model_noNANO(nn.Module):
    """
    Mamba model
    """

    def __init__(self, model_config: ModelConfig, dropout=0.1):
        """
        Embedder and multilayer model using the Mamba_block
        """

        super().__init__()
        self.d_model = model_config.embedding_dimension

        self.dropout = nn.Dropout(dropout)

        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedder = embedders.STD_embedder(model_config)

        # multi-layers transformer blocks, deep network
        config = MambaConfig(d_model=model_config.embedding_dimension, 
                             n_layers=model_config.num_layers,
                             expand_factor=model_config.forward_expansion,
                             d_state=model_config.d_state,
                             d_conv=4,
                             use_cuda=True )
        self.mamba_layers = Mamba(config)

    def forward(self, x, mask = None):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)
        
        x = self.dropout(x)

        # running over multiple transformer blocks
        x = self.mamba_layers(x)
        return x

class Nano_Bert_Differential_Efficient(nn.Module):
    """
    BERT model with NanoBERT embedder and efficient attention
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
                            blocks.EfficientDifferentialAttention(model_config.embedding_dimension, dropout), 
                            model_config.embedding_dimension, 
                            model_config.forward_expansion, 
                            dropout) 
                                    for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = modules.make_score_mask(mask)
        
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x
    



class Nano_Bert_Differential_Skip(nn.Module):
    """
    BERT model with NanoBERT embedder and efficient attention
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
            [
                blocks.EfficientDifferentialSkipAttention(model_config.embedding_dimension, 
                                                          model_config.d_state,
                                                          model_config.forward_expansion,
                                                          dropout) 
                
                for _ in range(model_config.num_layers)])

    def forward(self, x, mask):
        
        # embedding the indexed sequence to sequence of vectors
        x = self.embedder(x)

        # attention masking for padded token
        # (batch_size, seq_len, seq_len)
        mask = modules.make_score_mask(mask)
        
        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x