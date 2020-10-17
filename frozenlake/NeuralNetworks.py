import torch


class LinearNetworkWithTwoHiddenLayers(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_nodes_l1=100, hidden_nodes_l2=50, learning_rate=0.001):
        super(LinearNetworkWithTwoHiddenLayers, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_space, hidden_nodes_l1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes_l1, hidden_nodes_l2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes_l2, action_space))
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)


class LinearNetworkWithOneHiddenLayer(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_nodes=100, learning_rate=0.001):
        super(LinearNetworkWithOneHiddenLayer, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_space, hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes, action_space))
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)
