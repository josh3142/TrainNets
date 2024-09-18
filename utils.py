import torch
import random
import numpy as np


def make_deterministic(seed: int) -> None:
    random.seed(seed)   	    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)