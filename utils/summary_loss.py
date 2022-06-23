import torch
import numpy as np


class SummaryLoss(torch.nn.Module):
    """ Basic loss wrapper that calcualtes multihead summary loss """

    def __init__(self, loss: torch.nn.Module, loss_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            loss (torch.nn.Module): Loss function (math) used for multihead summary loss calculation (e.g. nn.CrossEntropyLoss)
        """
        super().__init__()
        self.loss = loss()
        self.loss_weights = loss_weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = []
        loss_sum = 0
        loss_weight_sum = 0
        for prediction, target, loss_weight in zip(predictions, targets, self.loss_weights):
            one_loss = self.loss(prediction.to("cpu"), target.to("cpu"))
            losses.append(one_loss)
            loss_sum = loss_sum + (np.log(loss_weight)*one_loss)
            loss_weight_sum = loss_weight_sum + loss_weight
        loss_sum = loss_sum/loss_weight_sum    
        return loss_sum
