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
            nn.Linear(self.dim_in, self.dim_hidden * 2, bias=False),
            nn.LayerNorm(self.dim_hidden * 2, elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(self.dim_hidden * 2, self.dim_hidden, bias=False),
            nn.LayerNorm(self.dim_hidden, elementwise_affine=True),
            nn.ReLU(),
            nn.Linear(self.dim_hidden, self.dim_out, bias=False),
        )

        self.activ_layers = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}
        self.activation_name = activation_name

        # run weight initialization
        self.apply(self._init_weights)

    def forward(self, x):
        output = self.model(x)
        if self.activation_name not in self.activ_layers.keys():
            pass
        else:
            final_activ_func = self.activ_layers[self.activation_name]()
            output = final_activ_func(output)
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        elif isinstance(module, nn.Linear):
            module.weight = nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

