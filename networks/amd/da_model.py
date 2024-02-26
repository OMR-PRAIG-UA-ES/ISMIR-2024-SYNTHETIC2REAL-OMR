from copy import deepcopy

import torch

from networks.base.modules import BN_IDS
from networks.amd.da_loss import AMDLoss
from networks.base.model import CTCTrainedCRNN


class DATrainedCRNN(CTCTrainedCRNN):
    def __init__(self, src_checkpoint_path, ytest_i2w, bn_ids):
        # Save hyperparameters
        self.save_hyperparameters()
        # Source model checkpoint path
        self.src_checkpoint_path = src_checkpoint_path
        # Target dictionary
        self.ytest_i2w = ytest_i2w
        # BN identifiers
        assert all(
            [bn_id in BN_IDS for bn_id in bn_ids]
        ), f"bn_ids must be a subset of {BN_IDS}"
        self.bn_ids = bn_ids
        # Initialize source model
        self.initialize_src_model()
        # Predictions
        self.Y = []
        self.YHat = []
        # Summary
        self.summary()

    def initialize_src_model(self):
        # 1) Load source model
        print(f"Loading source model from {self.src_checkpoint_path}")
        src_model = CTCTrainedCRNN.load_from_checkpoint(
            self.src_checkpoint_path, ytest_i2w=self.ytest_i2w
        )
        src_model.freeze()
        # 2) Freeze all the layers except for the layers previous
        # to the batch normalization layers indicated by self.bn_ids
        # and save the running mean and variance of the batch normalization layers
        self.bn_mean = []
        self.bn_var = []
        for id in self.bn_ids:
            src_model.model.cnn.backbone[id - 1].weight.requires_grad = True
            self.bn_mean.append(src_model.model.cnn.backbone[id].running_mean)
            self.bn_var.append(src_model.model.cnn.backbone[id].running_var)
        # 3) Deep copy the source model
        self.model = deepcopy(src_model.model)
        # 4) Deep copy the source model's dictionaries
        self.w2i = deepcopy(src_model.w2i)
        self.i2w = deepcopy(src_model.i2w)
        # 5) Delete the source model
        for src_param, tgt_param in zip(
            src_model.model.parameters(), self.model.parameters()
        ):
            assert torch.all(
                torch.eq(src_param, tgt_param)
            ), "Source model and target model parameters are not equal"
        del src_model
        print("Source model ready!")

    def configure_da_loss(
        self,
        lr=3e-4,
        align_loss_weight=1.0,
        minimize_loss_weight=1.0,
        diversify_loss_weight=1.0,
    ):
        # 1) Initialize the learning rate
        self.lr = lr
        # 2) Initialize the loss function
        self.compute_da_loss = AMDLoss(
            bn_mean=self.bn_mean,
            bn_var=self.bn_var,
            align_loss_weight=align_loss_weight,
            minimize_loss_weight=minimize_loss_weight,
            diversify_loss_weight=diversify_loss_weight,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x = batch
        yhat, y_prev_bn = self.model.da_forward(x, self.bn_ids)
        loss_dict = self.compute_da_loss(yhat, y_prev_bn)
        for k, v in loss_dict.items():
            self.log(f"train_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        return loss_dict["loss"]
