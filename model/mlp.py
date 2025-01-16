import torch
from torch import nn, Tensor

from typing import Callable, Optional

class Sine(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)
    

class MLP(nn.Module):
    """
    Multi Layer Perceptron for image classification with the structure
    Architecture:
        1. x = flatten(x) # flatten the data to make it adaptable for MLP
        2. x = linear(x.size, n_hidden)(x)
        3. for n range(n_layer - 1)
        4. x = linear(n_hidden, n_hidden)(x)
        5. y = linear(n_hidden, y_size)
    Args:
        n_hidden: number of hidden units per layer
        n_layer: number of hidden layers
        H: height dimension 
        W: width dimension
        C: colour dimension
        n_class: number of output classes
        activation: Type of activation function
        init_var_y: Initialized variance for the output. If `init_var_y==0` 
            no variance for the output is learned
    """

    def __init__(
            self, 
            n_hidden: int, 
            n_layer: int, 
            H: int=1,
            W: int=1,
            C: int=1,
            n_class: int=1, 
            activation: Optional[str]="relu", 
            init_var_y: int=0,
            **kwargs
        ):
        super().__init__()
        self.n_input  = H * W * C
        self.n_class  = n_class
        self.n_hidden = n_hidden
        self.n_layer  = n_layer
        self.init_var_y = init_var_y

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sine":
            self.activation = Sine()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise NotImplementedError()
        
        # initialize variance as a positive number
        if init_var_y < 0:
            raise ValueError("Variance has to be initialized with a positive "
                f"number, but it is {init_var_y}.")
        elif init_var_y > 0:
            InverseSoftplus = lambda sigma: torch.log(torch.exp(sigma) - 1 )
            self.variance_param = nn.Parameter(
                InverseSoftplus(torch.tensor([init_var_y]))
            )
        
        self.features     = self._make_layers(self.n_input, self.n_hidden, 
            self.n_layer, self.activation)
        self.logit        = nn.Linear(self.n_hidden, self.n_class)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.flatten(x, 1)
        if self.n_hidden != 0:
            x = self.features(x)
        logit = self.logit(x)
        return logit
    
    def _make_layers(
            self, n_input: int, n_hidden: int, n_layer: int, nonlinear: Callable
        ) -> nn.Module:
        layers = []
        for n_layer in range(n_layer):
            layer_lin   = nn.Linear(n_input, n_hidden)
            n_input = n_hidden
            if nonlinear is not None:
                layers += [layer_lin, nonlinear]
            else:
                layers += [layer_lin]
        return nn.Sequential(*layers)
    
    def get_variance(self):
        return nn.Softplus()(self.variance_param)
    

if __name__ == "__main__":
    x, y = 1, 1
    n_hidden=10
    n_layer=2

    model = MLP(n_hidden, n_layer, x, y)
    # n_para = count_parameters(model)
    # print(model)
    # print(f"Number of parameters is {n_para}.")