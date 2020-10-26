import torch.nn as nn
import torch.nn.functional as F


class FLModel(nn.Module):
    feautre_num = 68
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(self.feautre_num, 256)
        self.fc5 = nn.Linear(256, 14)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc5(x)
        output = F.log_softmax(x, dim=1)

        return output

class MLP(nn.Module):
    dim_in, dim_hidden, dim_out = 68, 64, 14
    def __init__(self):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(self.dim_in, self.dim_hidden)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout()
        self.bn = nn.BatchNorm1d(self.dim_hidden, )
        self.layer_hidden = nn.Linear(self.dim_hidden, self.dim_out)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)