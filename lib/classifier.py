import torch
import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, model, model_out_sz, n_labels):
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(model_out_sz, n_labels)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        #x = x.mean(dim=1)
        x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))
        #x = x[:,-1,:]
        x = self.act(x)
        x = self.fc(x)
        return x
    
class classifier_SP(nn.Module):
    def __init__(self, model, model_out_sz, sentence_len, n_labels):
        super().__init__()
        self.model = model
        self.norm = nn.LayerNorm(model_out_sz)
        self.fc = nn.Linear(model_out_sz, n_labels)
        self.fc_slen = nn.Linear(sentence_len, 1)

    def forward(self, x:torch.Tensor):
        x = self.model(x)
        x = self.norm(x)
        x = self.fc(x)
        x = x.transpose(1, 2)
        x = self.fc_slen(x)
        x = x.squeeze()
        return x
    