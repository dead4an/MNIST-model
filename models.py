# Imports 
import torch
import torch.nn as nn
import torch.nn.functional as F


# Models
class LinearModel(nn.Module):
    """Model based on fully-connected linear layers."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout_p: float=0.3) -> None:
        super(LinearModel, self).__init__()
        self.fc_input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc_hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc_output = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout_p)

        self.input_dim = input_dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape input from (batch_size, img_size, img_size)
        # to (batch_size, num_pixels)
        output = input.view(-1, self.input_dim)

        # linear -> dropout -> activation
        output = F.relu(self.dropout(self.fc_input(output)))
        output = F.relu(self.dropout(self.fc_hidden(output)))
        output = F.log_softmax(self.fc_output(output), dim=-1)
        
        return output
    

class ConvModel(nn.Module):
    """Model based on convolutional and fully-connected linear layers."""
    def __init__(self, out_channels: int, hidden_dim: int, input_dim: int, 
                 output_dim: int, dropout_p: float=0.3) -> None:
        super(ConvModel, self).__init__()
        self.conv_input = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3)
        self.fc_hidden = nn.Linear(in_features=4160, out_features=hidden_dim)
        self.fc_output = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.pool = nn.MaxPool1d(kernel_size=3)
        self.bn = nn.BatchNorm1d(num_features=hidden_dim, track_running_stats=False)

        self.input_dim = input_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Reshape input from (batch_size, num_channels, img_size, img_size)
        # to (batch_size, num_channels, num_pixels)
        output = input.view(-1, 1, self.input_dim)
    
        # conv -> pooling -> activation
        output = F.relu(self.pool(self.conv_input(output)))
        output = output.view(output.size(0), -1)

        # linear -> batchnorm -> activation
        output = F.relu(self.bn(self.fc_hidden(output)))
        output = F.log_softmax(self.fc_output(output), dim=-1)

        return output
    