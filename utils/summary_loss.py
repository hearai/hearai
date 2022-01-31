import torch


class SummaryLoss(torch.nn.Module):
    """ Basic loss wrapper that calcualtes multihead summary loss """

    def __init__(self, loss):
        """
        Args:
            loss (torch.nn.Module): Loss function used for multihead summary loss calculation (e.g. nn.CrossEntropyLoss)
        """
        super().__init__()
        self.loss = loss()

    def forward(self, predictions, targets):
        losses = []
        loss_sum = 0
        for prediction, target in zip(predictions, targets):
            one_loss = self.loss(prediction.to("cpu"), target.to("cpu"))
            losses.append(one_loss)
            loss_sum = loss_sum + one_loss
        return loss_sum
