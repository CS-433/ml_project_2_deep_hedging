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


class MLP_debug(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, activation_name="ReLU"):
        super(MLP_debug, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out

        self.model = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hidden),
            nn.ReLU(),
            nn.LayerNorm(self.dim_hidden, elementwise_affine=True),
            nn.Linear(self.dim_hidden, self.dim_hidden * 2),
            nn.ReLU(),
            nn.LayerNorm(self.dim_hidden * 2, elementwise_affine=True),
            nn.Linear(self.dim_hidden * 2, self.dim_out, bias=False),
        )

        self.fc1 = nn.Linear(self.dim_in, self.dim_hidden)
        self.fc2 = nn.Linear(self.dim_hidden, self.dim_hidden * 2)
        self.fc3 = nn.Linear(self.dim_hidden * 2, self.dim_out, bias=False)

        self.ln1 = nn.LayerNorm(self.dim_hidden, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(self.dim_hidden * 2, elementwise_affine=True)

        self.activ_layers = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}
        self.activation_name = activation_name

        # run weight initialization
        self.apply(self._init_weights)

    def forward(self, x, isPrint=False):
        output = self.model(x)
        x = self.fc1(x)
        # print('after FC1 :',x)
        x = nn.ReLU()(x)
        # print('after ReLU :',x)
        x = self.ln1(x)
        # print('after layer norm :', x)

        x = self.fc2(x)
        # print('after FC2 :',x)
        x = nn.ReLU()(x)
        # print('after ReLU :',x)
        x = self.ln2(x)
        # print('after layer norm :', x)
        x = self.fc3(x)

        if self.activation_name not in self.activ_layers.keys():
            pass
        else:
            final_activ_func = self.activ_layers[self.activation_name]()
            output = final_activ_func(output)

        if isPrint:
            print(
                f"Ouputs - FC3: {round(x.item(),4)}, output: {round(output.item(),4)}"
            )
        return output

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            module.weight = nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
