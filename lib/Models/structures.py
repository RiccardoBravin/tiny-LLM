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

        self.ff = nn.Sequential(
            nn.Linear(d_model, int(ff_expansion*d_model)),
            nn.SiLU(),
            nn.Linear(int(ff_expansion*d_model), d_model),
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        """
        Args:
            embeddings: the input tensor of shape (batch_size, max_len, d_model)
            mask: the mask tensor of shape (batch_size, max_len) that contains 0 for padding tokens and 1 for the rest
        Returns:
            Tensor of shape (batch_size, max_len, d_model)
        """

        #applying pre_LN as it seems to be better for training and gradients
        x = self.layernorm1(embeddings)
        attended = self.dropout(self.attention(x, x, x, mask))

        # residual layer (skip connection)
        skipped = attended + embeddings

        x = self.layernorm2(skipped)
        # bottleneck with skip connection
        feed_forward_out = self.dropout(self.ff(x))

        encoded = feed_forward_out + skipped
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
    def __init__(self, d_model, hid_d_model):
        super().__init__()

        self.act = torch.nn.SiLU()

        #firs stage
        self.norm = modules.RMSNorm(d_model) #MAYBE USE RMS NORM???

        #second 1 stage
        self.d_conv = 16
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model,
                            kernel_size=self.d_conv,
                            groups=d_model,
                            padding=self.d_conv - 1)

        self.attention = blocks.EmbBertAttention(d_model, hid_d_model)


        #second 2 stage
        self.fc_up = nn.Linear(d_model, hid_d_model)
        
        #final aggregation
        self.fc_down = nn.Linear(hid_d_model, d_model)




    def forward(self, embeddings, mask = None):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        #first stage
        l = embeddings.shape[1]
        x = self.norm(embeddings)

        #second 1 stage
        y1 = x.transpose(1, 2)
        y1 = self.conv1d(y1)[:,:,:l] #could try to make value skip the convolution so in attention use the original embeddings
        y1 = y1.transpose(1, 2)
        y1 = self.act(y1)

        y1 = self.attention(y1, y1, y1, mask)


        #second 2 stage
        y2 = self.fc_up(x)
        y2 = self.act(y2)

        #final stage
        y = self.fc_down(y1 + y2)


        return y


class Att_idea1(nn.Module):
    def __init__(self, d_model, hid_d_model, s_len):
        super().__init__()

        self.norm = modules.RMSNorm(d_model)

        self.fc1 = nn.Linear(d_model, hid_d_model)
        self.fc2 = nn.Linear(d_model, hid_d_model)

        self.A = nn.Parameter(torch.randn(s_len, hid_d_model, hid_d_model))

        self.softmax = torch.nn.Softmax(dim=-1)
        self.act = torch.nn.Softplus()
    
    def forward(self, embeddings, mask = None):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        x = self.norm(embeddings)

        y1 = self.fc1(x)
        y2 = self.fc2(x)

        y1 = torch.einsum("blr, lrx -> blr", y1, self.A)
        y1 = torch.einsum("blr, lrx -> blr", y1, self.A)

        y2 = self.softmax(y2)

        y2 = torch.einsum("bld, blr -> brd", embeddings, y2)

        y = y1 @ y2

        #y = self.act(y)

        return y + embeddings


class NewLayer(nn.Module):
    def __init__(self, d_model, d_ff, hid_d_model, s_len):
        super().__init__()
        
        self.norm = modules.RMSNorm(d_model)


        self.att = blocks.NewPosAttention(d_model, s_len, hid_d_model)
        
        self.fcup = nn.Linear(d_model, d_ff)
        self.fcdown = nn.Linear(d_ff, d_model)
        
        self.fc = nn.Linear(d_model, d_model)

        self.act = torch.nn.SiLU()

    def forward(self, embeddings, mask = None):
        # embeddings: (batch_size, max_len, d_model)
        # result: (batch_size, max_len, d_model)

        x = self.att(embeddings, embeddings, embeddings, mask)
        
        y1 = self.fc(x)

        y2 = self.fcup(self.norm(x))
        y2 = self.act(y2)
        y2 = self.fcdown(y2)

        y = y1 * y2
        
        return y
    