import os 
import torch
from collections import OrderedDict

import hydra
from omegaconf import DictConfig

@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    path = "results/" + \
        f"{cfg.data.name}/{cfg.model.name}" + \
        f"/bs{cfg.optim.bs}lr{cfg.optim.lr}/seed{cfg.seed}/" + \
        f"lightning_logs/version_{cfg.ckpt.version}/checkpoints"
    
    files = os.listdir(path)
    for file in files:
        try:
            file = file.split(".")[0]
            state_dict = torch.load(
                os.path.join(path, f"{file}.ckpt"), 
                map_location="cpu"
            )["state_dict"]
            state_dict_rename_keys = OrderedDict(
                (key.replace("model.", ""), value) for key, value in state_dict.items()
            )
            torch.save(
                state_dict_rename_keys, 
                os.path.join(path, f"{file}_dict.ckpt")
            )
        except:
            pass

if __name__=="__main__":
    run_main()