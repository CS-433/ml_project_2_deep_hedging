import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)

class MLP(nn.Module):
    
    def __init__(self, dim_in, dim_hidden, dim_out, hidden_activ='ReLU', output_activ='ReLU'):
        super(MLP, self).__init__()
        activ_layers ={'ReLU': nn.ReLU, 'Sigmoid': nn.Sigmoid, 'Tanh': nn.Tanh, 'Swish':nn.SiLU}
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.model = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            activ_layers[hidden_activ](),
            nn.Linear(self.dim_hidden, self.dim_out),
            activ_layers[output_activ](),
        )

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, dim_out, kernel=2, stride=1, dropout: float = 0.1
    ):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                3,
                2,
            ),
            nn.ReLU(),
            nn.Linear(64, dim_out),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        return self.model(x)
