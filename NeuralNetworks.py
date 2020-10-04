import torch





class LinearNetworkWithOneHiddenLayer(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_nodes=64, learning_rate=1e-4):
        super(LinearNetworkWithOneHiddenLayer, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_space, hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_nodes, action_space))
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.model(x)
