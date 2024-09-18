import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import pandas as pd
import numpy as np

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from model.model import get_model
from dataset.dataset import get_dataset
from net import NetPred


def map_label_vector_to_one_hot_encoded_vector(
        n_class: int, 
        label_vector: Tensor
    ) -> Tensor:
    """ Converts label vector to one-hot encoded vector. """
    return torch.eye(n_class)[label_vector]

def get_accuracy(Y_hat: Tensor, Y: Tensor) -> float:
    """ 
    Computes the number of correct predictions.  
    Args:
        Y_hat: Either logits or softmax predictions of the model.
        Y: True classes
    """
    assert len(Y.shape) == 1
    Y_hat_correct = np.sum(np.equal(np.argmax(Y_hat, axis=1), Y)) / len(Y)

    return Y_hat_correct


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high") 

    path = "results/" + \
        f"{cfg.data.name}/{cfg.model.name}" + \
        f"/bs{cfg.optim.bs}lr{cfg.optim.lr}/seed{cfg.seed}"
    weights = os.path.join(
        path,
        f"lightning_logs/version_{cfg.ckpt.version}/checkpoints/{cfg.ckpt.name}"
    )

    # load model
    model = get_model(cfg.model.name, 
        **(dict(cfg.model.param) | dict(cfg.data.param)))
    model = NetPred.load_from_checkpoint(weights, 
        model=model, 
        optimizer=None,
        objective=None,
        is_classification=cfg.data.is_classification
    )
    model.eval()

   # initialize dataset and dataloader
    dl = DataLoader(
        get_dataset(cfg.data.name, cfg.data.path, train=False),
        shuffle=False,
        batch_size=cfg.optim.bs
    )

    # get true values
    Y_idcs =[]
    for _, y in dl:
        Y_idcs.append(y)
    Y_idcs = torch.cat(Y_idcs)
    if cfg.data.is_classification:
        Y = map_label_vector_to_one_hot_encoded_vector(cfg.data.param.n_class, Y_idcs)
    else:
        Y = Y_idcs

    # train model
    trainer = pl.Trainer(
        default_root_dir=path, 
        precision="32", 
        devices=cfg.device
    )
    with torch.no_grad():
        Y_hat = trainer.predict(model=model, dataloaders=dl)
        Y_hat = torch.cat(Y_hat)
    Y, Y_hat = Y.numpy(), Y_hat.numpy()
    assert Y.shape==Y_hat.shape, "Y and Y_hat need to have the same shape"

    
    # store dataframe
    n_col = Y.shape[-1]
    Y_name = [f"Y{i}" for i in range(n_col)]
    Y_hat_name = [f"Y_hat{i}" for i in range(n_col)]
    column_name = Y_name + Y_hat_name
    
    array = np.concatenate([Y, Y_hat], axis=-1)
    df = pd.DataFrame(array, columns=column_name)
    df.to_csv(os.path.join(path,"pred.csv"), index=False)

    if cfg.data.is_classification:
        with open(os.path.join(path, "accuracy.txt"), "w") as file:
            file.write(f"Accuracy: {get_accuracy(Y_hat, Y_idcs.numpy())}")


if __name__ == "__main__":
    run_main()