import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, SAGEConv


class ProtGNN(torch.nn.Module):
    def __init__(self, input_dim=25, hidden_channels=256, out_channels=128):
        super().__init__()
        
        self.conv1 = SAGEConv(input_dim, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        
        self.pool = global_max_pool
        
        self.lin1 = torch.nn.Linear(hidden_channels * 3, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1_pooled = self.pool(x1, batch)  # (batch_size, hidden_channels)
        
        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2_pooled = self.pool(x2, batch)  # (batch_size, hidden_channels)
        
        x3 = self.conv3(x2, edge_index)
        x3 = F.relu(x3)
        x3_pooled = self.pool(x3, batch)  # (batch_size, hidden_channels)
        
        x_combined = torch.cat([x1_pooled, x2_pooled, x3_pooled], dim=1)  # (batch_size, 3*hidden_channels)
        
        x = F.relu(self.lin1(x_combined))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        
        return x
    