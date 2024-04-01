import random

import torch
from torch.nn import CTCLoss
from torchinfo import summary
from lightning.pytorch import LightningModule


from networks.base.modules import CRNN
from my_utils.data_preprocessing import IMG_HEIGHT, NUM_CHANNELS
from my_utils.metrics import (
    compute_metrics,
    standard_ctc_greedy_decoder,
    split_ctc_greedy_decoder,
)


class CTCTrainedCRNN(LightningModule):
    def __init__(
        self,
        w2i: dict[str, int],
        i2w: dict[int, str],
        encoding_type: str = "standard",
    ):
        super(CTCTrainedCRNN, self).__init__()
        # Save hyperparameters
        self.save_hyperparameters()
        # Dictionaries
        self.w2i = w2i
        self.i2w = i2w
        # Model (we use the same token for padding and CTC-blank; w2i contains the token "<PAD>")
        self.model = CRNN(output_size=len(w2i))
        self.summary()
        # CTC Loss (we use the same token for padding and CTC-blank)
        self.blank_padding_token = w2i["<PAD>"]
        self.compute_ctc_loss = CTCLoss(
            blank=self.blank_padding_token, zero_infinity=True
        )
        # CTC Decoder
        self.encoding_type = encoding_type
        self.ctc_decoder = (
            standard_ctc_greedy_decoder
            if self.encoding_type == "standard"
            else split_ctc_greedy_decoder
        )
        # Predictions
        self.Y = []
        self.YHat = []

    def summary(self):
        summary(self.model, input_size=[1, NUM_CHANNELS, IMG_HEIGHT, 256])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        x, xl, y, yl = batch
        yhat = self.forward(x)
        # ------ CTC Requirements ------
        # yhat: [batch, frames, vocab_size]
        yhat = yhat.log_softmax(dim=-1)
        yhat = yhat.permute(1, 0, 2).contiguous()
        # ------------------------------
        loss = self.compute_ctc_loss(yhat, y, xl // self.model.width_reduction, yl)
        self.log("train_ctc_loss", loss, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def validation_step(self, batch: list[torch.Tensor, tuple[str]]):
        # Model prediction
        x, y = batch  # batch_size = 1
        yhat = self.forward(x)[0]
        yhat = yhat.log_softmax(dim=-1).detach().cpu()
        yhat = self.ctc_decoder(
            y_pred=yhat, i2w=self.i2w, blank_padding_token=self.blank_padding_token
        )
        # Append to later compute metrics
        self.Y.extend(y)  # batch_size = 1
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
