# needed for deterministic behaviour
# https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
import os 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import StochasticWeightAveraging, ModelCheckpoint


import hydra 
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

from model.model import get_model
from dataset.dataset import get_dataset
from net import NetPred
from utils import make_deterministic, get_objective, get_objective_name

# Define a custom callback to delete older checkpoints
class KeepLastNCheckpoints(pl.Callback):
    def __init__(self, keep_last_n: int=20):
        super().__init__()
        self.keep_last_n = keep_last_n

    def on_train_epoch_end(self, 
            trainer: pl.Trainer, 
            pl_module: pl.LightningModule
        ) -> None:
        # Get all checkpoint files
        # path of checkpoints is infered from trainer
        if trainer.current_epoch > 0:
            self.checkpoint_dir = os.path.join(
                trainer.logger.log_dir, "checkpoints")
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) 
                                if f.endswith(".ckpt")]
            
            # Sort checkpoints by modification time
            checkpoint_files.sort(key=lambda f: os.path.getmtime(
                os.path.join(self.checkpoint_dir, f)))
            
            # If there are more than `keep_last_n` checkpoints, delete the oldest ones
            if len(checkpoint_files) > self.keep_last_n:
                old_checkpoints = checkpoint_files[:-self.keep_last_n]
                for old_checkpoint in old_checkpoints:
                    os.remove(os.path.join(self.checkpoint_dir, old_checkpoint))
                    print(f"Deleted old checkpoint: {old_checkpoint}")


@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision("high") 
    make_deterministic(cfg.seed)

    # if key is not in cfg structure add it with trivial value
    if not OmegaConf.select(cfg.model.param, "init_var_y"):
        OmegaConf.update(
            cfg, "model.param.init_var_y", 0, merge=True, force_add=True
        )

    path = "results/" + \
        f"{cfg.data.name}/{cfg.model.name}" + \
        f"/bs{cfg.optim.bs}lr{cfg.optim.lr}/seed{cfg.seed}"
    Path(path).mkdir(parents=True, exist_ok=True)
    
    
    # initialize predictive model class
    model = get_model(cfg.model.name, 
        **(dict(cfg.model.param) | dict(cfg.data.param)))

    # initialize objective, optimizer and net class,
    init_lr   = cfg.optim.lr * cfg.optim.bs / 256   
    objective = get_objective(get_objective_name(
        cfg.data.is_classification, 
        train_variance=False if cfg.model.param.init_var_y==0 else True
        ))

    optimizer = torch.optim.Adam(model.parameters(), init_lr,
            betas        = (cfg.optim.adam.beta1, cfg.optim.adam.beta2),
            eps          = cfg.optim.adam.eps,
            weight_decay = cfg.optim.wd)

    # load checkpoint to continue training from given checkpoint
    try:
       # this only works if the hyperparameter are stored correctly
       model = NetPred.load_from_checkpoint(
            checkpoint_path=f"{path}/lightning_logs/version_/" +
            f"{cfg.ckpt.version}/checkpoints/{cfg.ckpt.name}")

    except:
        weights = os.path.join(
            path,
            f"lightning_logs/version_{cfg.ckpt.version}/checkpoints/{cfg.ckpt.name}")
        try:
            var_y = model.get_variance().item()
        except:
            print("Model has no variance stored. No, variance is used.")
            var_y = 0
        model = NetPred.load_from_checkpoint(weights, 
            model=model, 
            optimizer=optimizer,
            objective=objective,
            scheduler=None,
            is_classification=cfg.data.is_classification,
            init_var_y=var_y
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

    # should swa be applied
    if cfg.swa.enable:
        swa_callback = StochasticWeightAveraging(
                swa_lrs=cfg.swa.lr,
                swa_epoch_start=cfg.swa.epoch.start,
                annealing_epochs=cfg.swa.epoch.annealing
                )
        ckpt_callback = ModelCheckpoint(
            dirpath=None, # None infers the path from the trainer
            filename="{epoch}-swa",
            save_top_k=-1,
            save_weights_only=True,
            every_n_epochs=cfg.swa.epoch.frequ_to_save
        )
        callback = [
            swa_callback, 
            ckpt_callback, 
            KeepLastNCheckpoints(keep_last_n=cfg.swa.n_save)]
    else:
        callback = []

    # train model
    trainer = pl.Trainer(default_root_dir=path, precision="32", 
        devices=cfg.device, 
        max_epochs=cfg.epoch.end,
        check_val_every_n_epoch=1,
        callbacks=callback
    )
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_test)
    # store hyperparameters
    with open(os.path.join(trainer.logger.log_dir, "hyperparameters.json"), "w") as file:
        json.dump(OmegaConf.to_container(cfg, resolve=True), file, indent=4)

if __name__ == "__main__":
    run_main()