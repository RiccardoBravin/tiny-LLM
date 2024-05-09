import torch
import torch.nn as nn

    

class Embedder(torch.nn.Module):

    def __init__(self, vocab_size, embed_size, reduced_embed_sz = 16, seq_len=128):
        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, reduced_embed_sz, padding_idx=0)
        self.expander = nn.Linear(reduced_embed_sz, embed_size)
        self.position = torch.nn.Embedding(seq_len, embed_size)
       
    def forward(self, sequence):
        #build the positions tensor of the tokens
        position = torch.arange(sequence.size(1)).unsqueeze(0).expand_as(sequence).to(sequence.device)

        x = self.expander(self.token(sequence)) + self.position(position)
        return x

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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
    


class MLPSwiGLU(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, sentence_length, fw_expand, dropout=0.1):
        super().__init__()
        self.embededer = Embedder(vocab_size, d_model, seq_len=sentence_length)
        self.mlp_layers = torch.nn.ModuleList([SwiGLU(d_model, d_model*fw_expand) for _ in range(n_layers)])
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        
        x = self.embededer(x)
        for layer in self.mlp_layers:
            x = layer(x)
            x = self.dropout(x)
        return x