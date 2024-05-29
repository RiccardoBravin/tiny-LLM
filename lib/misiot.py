import torch
import torch.nn as nn

class Embedder(torch.nn.Module):

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
    
class Mixer(torch.nn.Module):
    def __init__(self, model1, model2, model3, vocab_size, embed_size, reduced_embed_sz, seq_len=128, dropout=0.1, parallel = True):

        super().__init__()

        self.parallel = parallel
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.embedder = Embedder(vocab_size, embed_size, reduced_embed_sz, seq_len, dropout)
        
    def forward(self, x):
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        
        x = self.embedder(x)
        
        if self.parallel:
            x1 = self.model1(x, mask)
            x2 = self.model2(x, (x > 0))
            x3 = self.model3(x)
            y = x1 + x2 + x3
        else:
            x1 = self.model1(x,  mask)
            x2 = self.model2(x1, (x > 0))
            x3 = self.model3(x2)
            y = x3

        return y