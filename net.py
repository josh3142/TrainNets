import torch
from torch import Tensor, nn, optim

import lightning.pytorch as pl

from typing import Tuple, Optional


class NetPred(pl.LightningModule):
    """ Classification network class. """

    def __init__(self, 
        model: nn.Module, 
        optimizer: Optional[optim.Optimizer], 
        objective: Optional[nn.Module],
        scheduler: Optional[optim.lr_scheduler],
        is_classification: bool=True,
        init_var_y: int=0
        ) -> None:
        """
        None should only be used, if a subset of methods is relevant that don't
        need the respective attribute.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.objective = objective if objective is not None else None
        self.scheduler = scheduler
        self.is_classification = is_classification
        self.y_variance = init_var_y
        self.save_hyperparameters()
        

    def __call__(self, *args) -> nn.Module:
        return self.model(*args)
    
    
    def configure_optimizers(self):
        if self.scheduler is None:
            return {
                'optimizer': self.optimizer,
                'monitor': 'val_loss'
            }
        else:
            return {
                'optimizer': self.optimizer,
                'lr_scheduler': {
                    "scheduler": self.scheduler,
                    "interval": "step", # Update learning rate every step (batch)
                    "frequency": 1 # This ensures the scheduler is stepped every batch
                },
                'monitor': 'val_loss'
            }

    
    def get_loss(self, logit, Y):
        if self.y_variance==0:
            loss = self.objective(logit, Y)
        else:
            bs, n_c = logit.shape
            self.y_variance = self.model.get_variance()
            variance = self.y_variance.expand(bs, n_c)
            assert variance.shape==logit.shape, "Variance has wrong shape."
            loss = self.objective(logit, Y, variance)
        return loss


    def get_accuracy(self, Y_hat: Tensor, Y: Tensor) -> float:
        """ 
        Computes the number of correct predictions.  
        Args:
            Y_hat: Either logits or softmax predictions of the model.
            Y: True classes
        """
        assert len(Y.shape) == 1
        Y_hat_correct = torch.sum(torch.argmax(Y_hat, dim = 1) == Y) / len(Y)

        return Y_hat_correct


    def training_step(self, batch: Tuple, batch_idx: int) -> float:
        X, Y = batch
        logit = self.model(X)
        loss = self.get_loss(logit, Y)
        self.log("train_loss", loss)
        if self.is_classification:
            accuracy = self.get_accuracy(logit, Y)
            self.log("train_acc", accuracy)
        # log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr)
        # log variance of output y if it is used
        if not self.y_variance==0:
            self.log("train_y_variance", self.y_variance)

        return loss


    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        X, Y = batch
        logit = self.model(X)
        loss = self.get_loss(logit, Y)
        self.log("val_loss", loss)
        if self.is_classification:
            accuracy = self.get_accuracy(logit, Y)
            self.log("val_acc", accuracy)
        if not self.y_variance==0:
            self.log("val_y_variance", self.y_variance)


    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        X, Y = batch
        logit = self.model(X)
        loss = self.get_loss(logit, Y)
        self.log("test_loss", loss)

    
    def predict_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """ Returns true value and predictions """
        X, Y = batch
        logit = self.model(X)
        if self.is_classification:
            logit = nn.functional.softmax(logit, dim=-1)
        return logit
