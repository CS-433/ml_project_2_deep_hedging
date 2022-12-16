import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, activation_name="ReLU"):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.model = nn.Sequential(
            nn.LayerNorm(self.dim_in, elementwise_affine=True),
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.ReLU(),
            nn.LayerNorm(self.dim_hidden, elementwise_affine=True),
            nn.Linear(self.dim_hidden, self.dim_hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(self.dim_hidden * 2, elementwise_affine=True),
            nn.Linear(self.dim_hidden * 2, self.dim_out),
        )

        self.activ_layers = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}
        self.activation_name = activation_name

    def forward(self, x):
        output = self.model(x)
        if self.activation_name not in self.activ_layers.keys():
            pass
        else:
            final_activ_func = self.activ_layers[self.activation_name]()
            output = final_activ_func(output)
        return output
