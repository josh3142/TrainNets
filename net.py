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
        is_classification: bool=True
        ) -> None:
        """
        None should only be used, if a subset of methods is relevant that don't
        need the respective attribute.
        """
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.objective = objective if objective is not None else None
        self.is_classification = is_classification
        self.save_hyperparameters()
        

    def __call__(self, *args) -> nn.Module:
        return self.model(*args)
    

    def configure_optimizers(self):
        return self.optimizer
    

    def get_loss(self, logit: Tensor, Y: Tensor) -> float:
        return self.objective(logit, Y)


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

        return loss


    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        X, Y = batch
        logit = self.model(X)
        loss = self.get_loss(logit, Y)
        self.log("val_loss", loss)
        if self.is_classification:
            accuracy = self.get_accuracy(logit, Y)
            self.log("val_acc", accuracy)


    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        loss = self.get_loss(batch)
        self.log("test_loss", loss)

    
    def predict_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """ Returns true value and predictions """
        X, Y = batch
        logit = self.model(X)
        if self.is_classification:
            logit = nn.functional.softmax(logit, dim=-1)
        return logit
