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

class Classifier_max(nn.Module):

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

        x = torch.max(x, dim=1).values

        x = self.act(x)
        x = self.fc(x)
        return x


class Classifier_BERT_pretraining(nn.Module):

    def __init__(self, model:nn.Module, model_out_sz: int, dictionary_size:int, num_labels:int):
        r"""
        Classifier that takes only the firts token of the sequence to classify the data
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model

        self.lm_cls = nn.Linear(model_out_sz, dictionary_size)


        self.class_cls = nn.Sequential(
            nn.Tanh(),
            nn.Linear(model_out_sz, num_labels),
        )

        self.lm_cls.weight = self.model.embedder.token.weight

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)


        y1 = self.lm_cls(x)

        x = x[:,0,:]
        y2 = self.class_cls(x)

        return y1, y2


class Classifier_Nano_BERT_pretraining(nn.Module):

    def __init__(self, model:nn.Module, model_out_sz: int, reduced_embedding_dimension: int,  dictionary_size:int, num_labels:int):
        r"""
        Classifier that takes only the firts token of the sequence to classify the data
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model

        self.reducer = nn.Linear(model_out_sz, reduced_embedding_dimension, bias=False)
        self.lm_cls = nn.Linear(reduced_embedding_dimension, dictionary_size)

        self.class_cls = nn.Sequential(
            nn.Tanh(),
            nn.Linear(model_out_sz, num_labels),
        )

        self.reducer.weight = torch.nn.Parameter(self.model.embedder.tok_expander.weight.T)
        self.lm_cls.weight = self.model.embedder.token.weight

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)

        y1 = self.reducer(x)
        y1 = self.lm_cls(y1)

        x = x[:,0,:]
        y2 = self.class_cls(x)

        return y1, y2


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
    
class Classifier_first_token(nn.Module):

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
        self.act = nn.Tanh()
        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        x = x[:,0]
        x = self.act(x)
        x = self.fc(x)
        return x    


#REALLY BAD
class Smart_classifier(nn.Module):
    def __init__(self, model:nn.Module, model_out_sz: int, hidden_state:int, labels_num:int):
        r"""
        Final classifier that differs from usual ones that either take the root mean square of the embeddings or the first token.
        This classifier implies a new layer that merges informations from all the embeddings in the sequence and then with a fc layer classifies the data
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout(0.1)

        self.fc_delta1 = nn.Linear(model_out_sz, 1)
        self.fc_delta2 = nn.Linear(1, model_out_sz)

        nn.init.uniform_(self.fc_delta2.weight, -1, 1)
        dt = torch.exp(torch.rand(model_out_sz) * 4.6 - 6.9).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.fc_delta2.bias.copy_(inv_dt)

        self.fc_B = nn.Linear(model_out_sz, hidden_state)

        A = torch.arange(1, hidden_state + 1).repeat(model_out_sz,1)

        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.norm = nn.LayerNorm(hidden_state)

        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):

        x = self.model(x, mask)
        x = self.dropout(x)

        b, l, d_in = x.size()
        n = self.A_log.shape[1]

        A = -torch.exp(self.A_log.float())

        delta = self.fc_delta1(x)
        B = self.fc_B(x)

        delta = torch.nn.functional.softplus(self.fc_delta2(delta)) #(batch, seq_len, d_model)

        deltaA = torch.exp(torch.einsum('b l d, d n -> b l d n', delta, A)) #(batch, seq_len, d_model, hidden_state)
        deltaB_x = torch.einsum('b l d, b l n, b l d -> b l d n', delta, B, x) #(batch, seq_len, d_model, hidden_state)

        h = torch.zeros((b, d_in, n), device=deltaA.device)

        for i in range(l):
            h = deltaA[:, i] * h + deltaB_x[:, i]

        y = self.fc(h.squeeze(2))

        return y


#VERY GOOD
class Conv_classifier(nn.Module):
    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier used for testing purposes
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.d_conv = 16

        self.conv1d = nn.Conv1d(in_channels=model_out_sz, out_channels=model_out_sz,
                              kernel_size=self.d_conv,
                              groups=model_out_sz,
                              padding=self.d_conv - 1)

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(model_out_sz, labels_num)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):
        x = self.model(x, mask)
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)

        x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))

        x = self.act(x)
        x = self.fc(x)
        return x


#NOT SO GOOD
class Conv_classifier_2(nn.Module):
    def __init__(self, model:nn.Module, model_out_sz: int, labels_num:int):
        r"""
        Classifier used for testing purposes
        Args:
            model: the model that will be used to generate the embeddings
            model_out_sz: the output size of the model
            labels_num: the number of labels to output
        """
        super().__init__()
        self.model = model
        self.d_conv = 16
        assert(model_out_sz % 16 == 0)

        self.conv1d = nn.Conv1d(in_channels=model_out_sz, out_channels=model_out_sz,
                              kernel_size=self.d_conv,
                              groups=model_out_sz,
                              stride=self.d_conv // 2,
                              padding=0)


        self.w = nn.Parameter(torch.ones(4))

        self.act = nn.Sigmoid()

        self.fc = nn.Linear(model_out_sz, labels_num)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x:torch.Tensor, mask:torch.Tensor):

        x = self.model(x, mask)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)
        while x.shape[-1] > self.d_conv:

            x = self.conv1d(x)
        x = x.transpose(1, 2)

        x = torch.sqrt(torch.mean(torch.pow(x, 2), dim=1))
        x = self.act(x)
        x = self.fc(x)
        return x
