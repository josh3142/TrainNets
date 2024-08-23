# plot correlation between true and predicted value.
import os

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 17}
matplotlib.rc('font', **font)

@hydra.main(config_path = "config", config_name = "config")
def run_main(cfg: DictConfig) -> None:

    path = "results/" + \
        f"{cfg.data.name}/{cfg.model.name}" + \
        f"/bs{cfg.optim.bs}lr{cfg.optim.lr}/seed{cfg.seed}"

    df = pd.read_csv(os.path.join(path, "pred.csv"), index_col=False)
    Y_names = df.columns

    for i in range(len(Y_names)//2):
        Y_name, Y_hat_name = Y_names[i], Y_names[len(Y_names)//2 + i]
        x_max = max(np.max(df[Y_name]), np.max(df[Y_hat_name])) 
        x = np.arange(x_max)
        
        plt.tight_layout()
        plt.title(r"Correlation of predictions $\hat{Y}$ to true value $Y$.")
        plt.scatter(df[Y_name], df[Y_hat_name], c="orange", marker="x", 
            label=fr"$\hat{{Y}}_{i}/Y_{i}$")
        plt.plot(x, x, color="black", linestyle="--", label="optimum")
        plt.xlabel(fr"$Y_{i}$", color="black")
        plt.ylabel(rf"$\hat{{Y}}_{i}$", color="black")
        plt.legend()

        plt.savefig(os.path.join(path, f"correlation_plot_{i}.png"), 
            transparent=False, bbox_inches="tight")
        plt.clf()

if __name__=="__main__":
    run_main()
