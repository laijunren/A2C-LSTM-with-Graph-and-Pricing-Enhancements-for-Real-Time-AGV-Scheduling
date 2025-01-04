import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ActorNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, lstm_hidden_size, out_size):
        super(ActorNetwork, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, out_size)

    def forward(self, x, edge_index, hidden):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = F.relu(x)
        x = x.unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)
        return x, hidden
