import torch
import torch.nn as nn


class AMDLoss(nn.Module):
    def __init__(
        self,
        bn_stats: dict[int, dict[str, torch.Tensor]],
        align_loss_weight: float = 1.0,
        minimize_loss_weight: float = 1.0,
        diversify_loss_weight: float = 1.0,
    ):
        super(AMDLoss, self).__init__()
        self.bn_stats = bn_stats
        self.align_loss_weight = align_loss_weight
        self.minimize_loss_weight = minimize_loss_weight
        self.diversify_loss_weight = diversify_loss_weight

    def forward(
        self, y: torch.Tensor, y_prev_bn: dict[int, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # 1) Align loss: minimize the KL divergence between the batch and the running stats!
        align_loss = 0.0
        for id, stat_dict in self.bn_stats.items():
            align_loss += self.compute_align_loss(
                x=y_prev_bn[id],
                bn_mean=stat_dict["mean"],
                bn_var=stat_dict["var"],
            )
        align_loss *= self.align_loss_weight
        # 2) Minimize loss: minimize the entropy of the time step -> one-hot vectors!
        minimize_loss = self.compute_minimize_loss(y)
        minimize_loss *= self.minimize_loss_weight
        # 3) Diversify loss: maximize the entropy of the avg. frame -> variance within the batch!
        diversify_loss = -1 * self.compute_diversify_loss(y)
        diversify_loss *= self.diversify_loss_weight
        # 4) Total loss
        loss = align_loss + minimize_loss + diversify_loss
        return {
            "amd_loss": loss,
            "align_loss": align_loss,
            "minimize_loss": minimize_loss,
            "diversify_loss": diversify_loss,
        }

    @staticmethod
    def compute_align_loss(
        x: torch.Tensor,
        bn_mean: torch.Tensor,
        bn_var: torch.Tensor,
    ) -> torch.Tensor:
        # x: [b, c, h, w] -> 4D output of a 2D convolutional layer

        # Compute the mean and variance of the current batch
        # We want those two to equal the running mean and running var stats tracked by the BN layer
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
    def entropy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # x: [batch, features] -> logits
        x = x.softmax(dim=1)
        x = torch.clamp(x, min=eps, max=1.0)  # Avoid zero probabilities
        h = -1 * ((x * torch.log(x)).sum(dim=1))
        h = h.mean()
        return h

    def compute_minimize_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, frames, features]
        B, T, F = x.shape
        x = x.reshape(B * T, F)  # Take each frame as a sample!
        loss = self.entropy(x)
        return loss

    def compute_diversify_loss(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, frames, features]
        x = x.mean(dim=0)  # [frames, features]
        loss = self.entropy(x)
        return loss
