import numpy as np
import torch
import torch.nn as nn

from my_utils.data_preprocessing import NUM_CHANNELS, IMG_HEIGHT


BN_IDS = [1, 5, 9, 13]


class CRNN(nn.Module):
    def __init__(self, output_size: int):
        super(CRNN, self).__init__()
        # Encoder
        self.encoder = CNN()
        # Decoder
        self.decoder_input_size = IMG_HEIGHT // self.encoder.height_reduction
        self.decoder_input_size *= self.encoder.out_channels
        self.decoder = RNN(input_size=self.decoder_input_size, output_size=output_size)
        # Constants
        self.height_reduction = self.encoder.height_reduction
        self.width_reduction = self.encoder.width_reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder (CNN)
        x = self.encoder(x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.decoder_input_size)
        # Decoder (RNN)
        x = self.decoder(x)
        return x

    def da_forward(
        self, x: torch.Tensor, bn_ids: list[int]
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        prev_norm = {}
        # Encoder (CNN)
        for id, layer in enumerate(self.encoder.backbone):
            if isinstance(layer, nn.BatchNorm2d) and id in bn_ids:
                prev_norm[id] = x.clone()
            x = layer(x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.decoder_input_size)
        # Decoder (RNN)
        x = self.decoder(x)
        return x, prev_norm


######################################################## Encoder (CNN):


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Configuration
        config = {
            "filters": [NUM_CHANNELS, 64, 64, 128, 128],
            "kernel": [5, 5, 3, 3],
            "pool": [[2, 2], [2, 1], [2, 1], [2, 1]],
            "leaky_relu": 0.2,
        }
        self.backbone = nn.Sequential()
        for i in range(len(config["filters"]) - 1):
            self.backbone.append(
                nn.Conv2d(
                    config["filters"][i],
                    config["filters"][i + 1],
                    config["kernel"][i],
                    padding="same",
                    bias=False,
                )
            )
            self.backbone.append(nn.BatchNorm2d(config["filters"][i + 1]))
            self.backbone.append(nn.LeakyReLU(config["leaky_relu"], inplace=True))
            self.backbone.append(nn.MaxPool2d(config["pool"][i]))

        # Constants
        self.height_reduction, self.width_reduction = np.prod(
            config["pool"], axis=0
        ).tolist()
        self.out_channels = config["filters"][-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


######################################################## Decoder (RNN):


class RNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(RNN, self).__init__()
        # Configuration
        self.blstm = nn.LSTM(
            input_size,
            256,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(256 * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
