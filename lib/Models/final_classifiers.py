import torch
from torch import nn

class Classifier_rms(nn.Module):

    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier that uses the root mean square of the embeddings along the sequence length to classify the data
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)

        x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))

        x = self.act(x)
        x = self.fc(x)
        return x
    

class Classifier_BERT(nn.Module):

    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier that takes only the firts token of the sequence to classify the data 
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        x = x[:,0,:]
        x = self.act(x)
        x = self.fc(x)
        return x
    
class Classifier_for_electra(nn.Module):
    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier that to be used for ELCTRA pretraining to ensure correct passage of variables through the model
        The model is expected to output a tensor of shape (batch_size, seq_len, 1)
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        x = self.act(x)
        x = self.fc(x)
        return x
    

class Classifier_post_electra(nn.Module):

    def __init__(self, model:nn.Module, seq_len: int, labels_num:int):
        r"""
        Classifier that takes advantage of the last layer of the ELECTRA pretraining model to classify the data
        The model is expected to output a tensor of shape (batch_size, seq_len, 1)
        Args:
            model: the model that will be used to generate the embeddings
            seq_len: the length of the sequence
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(seq_len, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        # squeeze the last dimension to make it work with the linear layer
        x = self.act(x.squeeze(-1))
        x = self.fc(x)
        return x
    

class Classifier_last_token(nn.Module):
    
    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier that takes only the last token of the sequence to classify the data 
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.act = nn.Sigmoid()
        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        x = x[:,-1,:]
        x = self.act(x)
        x = self.fc(x)
        return x