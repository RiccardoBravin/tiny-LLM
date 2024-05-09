import torch.nn as nn

class classifier(nn.Module):
    def __init__(self, model, model_out_sz, n_labels):
        super().__init__()
        self.model = model
        self.fc = nn.Linear(model_out_sz, n_labels)

    def forward(self, x):
        x = self.model(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x