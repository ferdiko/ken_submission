import torch
import torch.nn as nn

class RoutingModel(nn.Module):
    def __init__(self, num_models, hidden_sizes=[128, 64]):
        """
        Initialize the feed-forward neural network.

        Parameters:
            num_models (int): Number of models in the zoo
            hidden_sizes (list): A list of integers representing the number of neurons in each hidden layer.

        Input and output vector meaning:
            Input: one hot vector for each model if it has been invoked already + current prediction and certainty
            Output: Use output + one hot vector which models to route to if not use output
        """
        super(RoutingModel, self).__init__()

        layer_sizes = [num_models+2] + hidden_sizes + [num_models+1]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.softmax(self.layers[-1](x), dim=1)


class ThreshModel(nn.Module):
    def __init__(self, num_models, num_threshs=4, hidden_sizes=[128, 64]):
        """
        Initialize the feed-forward neural network.

        Parameters:
            num_models (int): Number of models in the zoo
            hidden_sizes (list): A list of integers representing the number of neurons in each hidden layer.

        Input and output vector meaning:
            Input: one hot vector for each model if it has been invoked already + current prediction and certainty
            Output: Use output + one hot vector which models to route to if not use output
        """
        super(ThreshModel, self).__init__()

        layer_sizes = [2*num_models + 2*num_threshs] + hidden_sizes + [num_threshs]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.softmax(self.layers[-1](x), dim=1)


class CombineModel(nn.Module):
    def __init__(self, num_models, num_methods, hidden_sizes=[128, 64]):
        """
        Initialize the feed-forward neural network.

        Parameters:
            num_models (int): Number of models in the zoo
            num_mehtods (int): Number of possible methods to combine the predictions
            hidden_sizes (list): A list of integers representing the number of neurons in each hidden layer.

        Input and output vector meaning:
            Input: one hot vector for each model if it has been invoked already + its certainty
            Output: method to use to combine the predictions
        """
        super(CombineModel, self).__init__()

        layer_sizes = [2*num_models] + hidden_sizes + [num_methods]
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes) - 1)])

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return torch.softmax(self.layers[-1](x))


# Define the Deep Q-Network (DQN) model
class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x