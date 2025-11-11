import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class MultiModalFusion(nn.Module):
    """
    A neural network module for fusing multiple modal embeddings using weighted combination.
    
    This module learns weights for different modalities and combines them through weighted sum
    after normalization.

    Attributes:
        - modal_num: Number of modalities to fuse
        - weight: Learnable weights for each modality
        - requires_grad: Whether the weights are trainable
    """

    def __init__(self, modal_num: int, with_weight: int = 1):
        """
        Initializes the MultiModalFusion module.

        Args:
            modal_num: Number of modalities to fuse
            with_weight: Whether to use learnable weights (1) or fixed weights (0)
        """
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)), requires_grad=self.requires_grad)

    def forward(self, embs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiModalFusion.

        Args:
            embs: Input tensor of shape (batch_size, modal_num, embedding_dim)
                 containing embeddings from different modalities

        Returns:
            torch.Tensor: Fused embedding tensor of shape (batch_size, embedding_dim)
        """
        assert embs.shape[1] == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        weight_norm = weight_norm.view(1, self.modal_num, 1)
        
        embs_normalized = F.normalize(embs, dim=2) 
        weighted_embs = embs_normalized * weight_norm
        joint_emb = torch.sum(weighted_embs, dim=1)
        
        return joint_emb


def get_activation_fn(activation):
    """Returns the activation function corresponding to `activation`"""

    if activation == "relu":
        return nn.ReLU() #F.relu
    elif activation == "gelu":
        return nn.GELU() #F.gelu
    elif activation == "elu":
        return nn.ELU()
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


class Projector(nn.Module):
    """
    A neural network module used for simple classification tasks. It consists of a two-layered linear network
    with a nonlinear activation function in between.

    Attributes:
        - linear1: The first linear layer.
        - linear2: The second linear layer that outputs to the desired dimensions.
        - activation_fn: The nonlinear activation function.
    """

    def __init__(
        self,
        input_dim,
        out_dim,
        activation_fn,
    ):
        """
        Initializes the Projector module.

        :param input_dim: Dimension of the input features.
        :param out_dim: Dimension of the output.
        :param activation_fn: The activation function to use.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation_fn = get_activation_fn(activation_fn)

    def forward(self, x):
        """
        Forward pass of the Projector.

        :param x: Input tensor to the module.

        :return: Tensor after passing through the network.
        """
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        return x


class Downsample_CNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()

        if input_dim // 4 > output_dim:
            hidden_dims = [input_dim // 2, input_dim // 4]
        elif input_dim * 4 < output_dim:
            hidden_dims = [input_dim * 2, input_dim * 4]
        else:
            hidden_dims = [input_dim // 2, input_dim // 2]

        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dims[0], kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.pool1 = nn.AdaptiveMaxPool1d(1)
        
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dims[0], out_channels=hidden_dims[1], kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        
        self.net3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dims[1], out_channels=output_dim*3, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.pool3 = nn.AdaptiveMaxPool1d(1)

        self.linear = nn.Linear(3 * output_dim, output_dim)
    
    def forward(self, x: Tensor):
        x = x.permute(0, 2, 1)
        x1 = self.net1(x)
        # mx1 = self.pool1(x1)
        # mx1 = mx1.squeeze(-1)

        x2 = self.net2(x1)
        # mx2 = self.pool2(x2)
        # mx2 = mx2.squeeze(-1)

        x3 = self.net3(x2)
        mx3 = self.pool3(x3)
        mx3 = mx3.squeeze(-1)
        
        # combined = torch.cat([mx1, mx2, mx3], dim=-1)
        out = self.linear(mx3)
        return out


class CNN_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_len = 1000):#, , device
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = 7
        self.do = nn.Dropout(0.1)
        # self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.fc = nn.Linear(self.input_dim, self.hidden_dim)
        self.ln = nn.LayerNorm(self.hidden_dim)
        self.convs = nn.ModuleList([nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2),
                                    nn.Conv1d(self.hidden_dim, self.hidden_dim*2, self.kernel_size, padding=(self.kernel_size-1)//2)])
        self.max_pool = nn.MaxPool1d(max_len)

    def forward(self, feat_map):
        h_map = self.fc(feat_map)
        h_map = h_map.permute(0, 2, 1)

        for i, conv in enumerate(self.convs):
            conved = conv(self.do(h_map))
            conved = F.glu(conved, dim=1)
            conved = (conved + h_map) * math.sqrt(0.5) #self.scale
            h_map = conved

        pool_map = self.max_pool(h_map).squeeze(-1)  # b,d
        h_map = h_map.permute(0, 2, 1)
        h_map = self.ln(h_map)    # b, len, d
        return h_map, pool_map


class MLP(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 1,
        act: str = 'gelu',  # relu, elu, none
        norm: str = 'layer',  # batch, none
        dropout_rate: float = 0.2
    ):
        super().__init__()
        layers = []
        if not hidden_dims:
            hidden_dims = [input_dim] # // 2
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if norm == 'layer':
                layers.append(nn.LayerNorm(dims[i+1]))
            elif norm == 'batch':
                layers.append(nn.BatchNorm1d(dims[i+1]))
            
            if act:
                layers.append(get_activation_fn(act))
            
            layers.append(nn.Dropout(dropout_rate))
        
        self.net = nn.Sequential(*layers)
    
        self.out_layer = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        x = self.net(x)
        x = self.out_layer(x)
        return x
