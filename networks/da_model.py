import random
from copy import deepcopy

import torch
from torchinfo import summary
from lightning.pytorch import LightningModule

from networks.modules import BN_IDS
from networks.model import CTCTrainedCRNN
from networks.da_loss import AMDLoss
from my_utils.data_preprocessing import NUM_CHANNELS, IMG_HEIGHT
from my_utils.metrics import ctc_greedy_decoder, compute_metrics


class DATrainedCRNN(LightningModule):
    def __init__(self, src_checkpoint_path, ytest_i2w, bn_ids):
        super(DATrainedCRNN, self).__init__()
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

    ############################################### Hereinafter:
    # Same as CTCTrainedCRNN
    # NOTE: But I'm having problems with the inheritance

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def forward(self, x):
        return self.model(x)

    def validation_step(self, batch, batch_idx):
        x, y = batch  # batch_size = 1
        # Model prediction (decoded using the vocabulary on which it was trained)
        yhat = self.model(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = ctc_greedy_decoder(yhat, self.i2w)
        # Decoded ground truth
        y = [self.ytest_i2w[i.item()] for i in y[0]]
        # Append to later compute metrics
        self.Y.append(y)
        self.YHat.append(yhat)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self, name="val", print_random_samples=False):
        metrics = compute_metrics(y_true=self.Y, y_pred=self.YHat)
        for k, v in metrics.items():
            self.log(f"{name}_{k}", v, prog_bar=True, logger=True, on_epoch=True)
        # Print random samples
        if print_random_samples:
            index = random.randint(0, len(self.Y) - 1)
            print(f"Ground truth - {self.Y[index]}")
            print(f"Prediction - {self.YHat[index]}")
        # Clear predictions
        self.Y.clear()
        self.YHat.clear()
        return metrics

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end(name="test", print_random_samples=True)
