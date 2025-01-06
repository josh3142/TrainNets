import torch
from torch import nn, Tensor
import random
import numpy as np

from typing import Callable


def make_deterministic(seed: int) -> None:
    random.seed(seed)   	    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_objective_name(is_classification: bool, train_variance: bool=False
    ) -> str:
    if is_classification:
        return "crossentropy"
    else:
        if not train_variance:
            return "mse"
        else:
            return "gaussiannll"

def get_objective(name: str) -> Callable:
    if name=="mse":
        return nn.MSELoss()
    elif name=="crossentropy":
        return nn.CrossEntropyLoss()
    elif name=="gaussiannll":
        return nn.GaussianNLLLoss()
    elif name=="multivariategaussiannll": # This is not implemented in net.py
        return multivariate_gaussian_ll
    else:
        raise NotImplementedError(f"Objective {name} is not implemented.")
    

def multivariate_gaussian_ll(
        predictions: Tensor, 
        targets: Tensor, 
        covariance: Tensor,
        mean: bool = True
    ) -> Tensor:
    """
    Compute the negative log-likelihood for a multivariate Gaussian distribution.
    
    Args:
        x (torch.Tensor): Data points, shape (batch_size, k)
        mean (torch.Tensor): Mean vector, shape (batch_size, k)
        covariance (torch.Tensor): Covariance matrix, shape (batch_size, k, k)
    
    Returns:
        torch.Tensor: Negative log-likelihood for each data point in the batch
    """
    k = predictions.shape[1]  # Dimensionality of the data
     
    # Compute the weigthed mse: (x - mean)^T @ cov_inv @ (x - mean)
    batch_inv = torch.vmap(torch.linalg.inv, in_dims=0)
    cov_inv = batch_inv(covariance) 
    error = predictions - targets  
    weighted_mse = torch.einsum('bi,bij,bj->b', error, cov_inv, error)
    
    # Compute the log determinant term
    cov_logdet = torch.logdet(covariance)  # Shape: (batch_size,)
    
    nll =  0.5 * (weighted_mse + cov_logdet  + k * np.log(2 * np.pi))
    if not mean:
        return nll
    else:
        return nll.mean()