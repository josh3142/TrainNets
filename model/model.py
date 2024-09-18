from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from model.mlp import MLP
from model.resnet9 import ResNet9
from model.cnn import cnn_small

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(name: str, **kwargs) -> nn.Module:
    if name.startswith("mlp"):
        model = MLP(**kwargs)
    elif name=="resnet9":
        model = ResNet9(**kwargs)
    elif name=="cnn_small":
        model = cnn_small(**kwargs)
    elif name=="resnet18":
        pretrain = kwargs["pretrain"]
        n_class = kwargs["n_class"]
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrain else None)
        model.fc = nn.Linear(in_features=512, out_features=n_class, bias=True)
    else:
        raise NotImplementedError(name)
    
    return model