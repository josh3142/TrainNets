# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from torch import nn

import hydra 
from omegaconf import DictConfig 
from pathlib import Path

from model.model import get_model
from dataset.dataset import get_dataset
from net import NetPred


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high") 

    path = "results/" + \
        f"{cfg.data.name}/{cfg.model.name}" + \
        f"/bs{cfg.optim.bs}lr{cfg.optim.lr}/seed{cfg.seed}"
    Path(path).mkdir(parents = True, exist_ok = True)
    
    # initialize predictive model class
    model = get_model(cfg.model.name, 
        **(dict(cfg.model.param) | dict(cfg.data.param)))

    # initialize objective, optimizer and net class,
    init_lr   = cfg.optim.lr * cfg.optim.bs / 256   
    objective = nn.CrossEntropyLoss() if cfg.data.is_classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), init_lr,
            betas        = (cfg.optim.adam.beta1, cfg.optim.adam.beta2),
            eps          = cfg.optim.adam.eps,
            weight_decay = cfg.optim.wd)

    model = NetPred(
        model=model,
        optimizer=optimizer,
        objective=objective,
        is_classification=cfg.data.is_classification
    )

    # initialize dataset and dataloader
   # get data
    dl_train = DataLoader(
        get_dataset(cfg.data.name, cfg.data.path, train=True), 
        shuffle=True,
        batch_size=cfg.optim.bs
    )
    dl_test = DataLoader(
        get_dataset(cfg.data.name, cfg.data.path, train=False),
        shuffle=False,
        batch_size=cfg.optim.bs
    )

    # train model
    trainer = pl.Trainer(default_root_dir=path, precision="32", 
        devices=cfg.device, 
        max_epochs=cfg.epoch.end,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1)
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_test)


if __name__ == "__main__":
    run_main()