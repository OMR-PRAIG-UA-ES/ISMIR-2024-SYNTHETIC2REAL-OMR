import torch
import torch.nn as nn


class AMDLoss(nn.Module):
    def __init__(
        self,
        bn_mean,
        bn_var,
        align_loss_weight=1.0,
        minimize_loss_weight=1.0,
        diversify_loss_weight=1.0,
    ):
        super(AMDLoss, self).__init__()
        self.bn_mean = bn_mean
        self.bn_var = bn_var
        self.align_loss_weight = align_loss_weight
        self.minimize_loss_weight = minimize_loss_weight
        self.diversify_loss_weight = diversify_loss_weight

    def forward(self, y, y_prev_bn):
        # 1) Align loss (matching BN statistics):
        # Minimize the KL divergence between the batch and the running stats!
        align_loss = 0.0
        for id in range(len(y_prev_bn)):
            align_loss = align_loss + self.compute_align_loss(
                y_prev_bn[id], self.bn_mean[id], self.bn_var[id]
            )
        align_loss = self.align_loss_weight * align_loss
        # 2) Minimize loss:
        # Minimize the entropy of the time step -> one-hot vectors!
        minimize_loss = self.minimize_loss_weight * self.compute_minimize_loss(y)
        # 3) Variance loss:
        # Maximize the entropy of the avg. frame -> variance within the batch!
        diversify_loss = self.diversify_loss_weight * (
            -1 * self.compute_diversify_loss(y)
        )
        # Total loss
        loss = align_loss + minimize_loss + diversify_loss
        return {
            "loss": loss,
            "align_loss": align_loss,
            "minimize_loss": minimize_loss,
            "diversify_loss": diversify_loss,
        }

    @staticmethod
    def compute_align_loss(x, bn_mean, bn_var):
        # x: [b, c, h, w] -> 4D output of a 2D convolutional layer

        # Compute the mean and variance of the current batch
        # We want those two to equal the running mean and running var stats
        # tracked by the batch normalization layer
        # https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py

        x_mean = x.mean(dim=[0, 2, 3])
        x_var = x.var(dim=[0, 2, 3])
        # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        loss = (
            torch.log(torch.sqrt(bn_var) / torch.sqrt(x_var))
            + ((x_var + (x_mean - bn_mean).pow(2)) / (2 * bn_var))
            - 0.5
        )
        loss = loss.mean()

        return loss

    @staticmethod
    def entropy(x, eps=1e-4):
        # x: [batch, features] -> logits
        x = x.softmax(dim=1)
        x = torch.clamp(x, min=eps, max=1.0)  # Avoid zero probabilities
        h = -1 * ((x * torch.log(x)).sum(dim=1))
        h = h.mean()
        return h

    def compute_minimize_loss(self, x):
        # x: [batch, frames, features]
        B, T, F = x.shape
        x = x.reshape(B * T, F)  # Take each frame as a sample!
        loss = self.entropy(x)
        return loss

    def compute_diversify_loss(self, x):
        # x: [batch, frames, features]
        x = x.mean(dim=0)  # [frames, features]
        loss = self.entropy(x)
        return loss
