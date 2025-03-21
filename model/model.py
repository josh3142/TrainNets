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

if __name__=="__main__":
    print("Parameters for MLP (128/2) (California): ", 
          count_parameters(MLP(n_hidden=128, n_layer=2, C=8, n_class=1)))
    print("Parameters for MLP (128/2) (ENB): ", 
          count_parameters(MLP(n_hidden=128, n_layer=2, C=8, n_class=2)))
    print("Parameters for MLP (128/2) (Naval Propulsion): ", 
          count_parameters((MLP(n_hidden=128, n_layer=2, C=14, n_class=2))))
    print("Parameters for MLP (128/2) (Redwine): ", 
          count_parameters(MLP(n_hidden=128, n_layer=2, C=11, n_class=1)))
    print("Parameters for CNN ((Fashion)MNIST): ", 
          count_parameters(cnn_small(C=1, n_class=10)))
    print("Parameters for ResNet9 (Cifar10): ", 
          count_parameters(ResNet9(C=3, n_class=10)))
    print("Parameters for ResNet18 (ImageNet10): ", 
          count_parameters(get_model("resnet18", pretrain=False, C=3, n_class=10)))