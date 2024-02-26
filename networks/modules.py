import numpy as np
import torch.nn as nn

from my_utils.data_preprocessing import NUM_CHANNELS, IMG_HEIGHT

BN_IDS = [1, 5, 9, 13]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.config = {
            "filters": [NUM_CHANNELS, 64, 64, 128, 128],
            "kernel": [5, 5, 3, 3],
            "pool": [[2, 2], [2, 1], [2, 1], [2, 1]],
            "leaky_relu": 0.2,
        }
        self.bn_ids = []

        layers = []
        for i in range(len(self.config["filters"]) - 1):
            layers.append(
                nn.Conv2d(
                    self.config["filters"][i],
                    self.config["filters"][i + 1],
                    self.config["kernel"][i],
                    padding="same",
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(self.config["filters"][i + 1]))
            layers.append(nn.LeakyReLU(self.config["leaky_relu"], inplace=True))
            layers.append(nn.MaxPool2d(self.config["pool"][i]))
            # Save BN ids
            self.bn_ids.append(len(layers) - 3)  # [1, 5, 9, 13]
        assert BN_IDS == self.bn_ids, "BN ids are not the same!"

        self.backbone = nn.Sequential(*layers)
        self.height_reduction, self.width_reduction = np.prod(
            self.config["pool"], axis=0
        )
        self.out_channels = self.config["filters"][-1]

    def forward(self, x):
        x = self.backbone(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()

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

    def forward(self, x):
        x, _ = self.blstm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x


class CRNN(nn.Module):
    def __init__(self, output_size):
        super(CRNN, self).__init__()
        # CNN
        self.cnn = CNN()
        # RNN
        self.rnn_input_size = self.cnn.out_channels * (
            IMG_HEIGHT // self.cnn.height_reduction
        )
        self.rnn = RNN(input_size=self.rnn_input_size, output_size=output_size)

    def forward(self, x):
        # CNN
        x = self.cnn(x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.rnn_input_size)
        # RNN
        x = self.rnn(x)
        return x

    def da_forward(self, x, bn_ids):
        # CNN
        bn = []
        for i in range(len(self.cnn.backbone)):
            if i in bn_ids:
                bn.append(x.clone())
            x = self.cnn.backbone[i](x)
        # Prepare for RNN
        b, _, _, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.reshape(b, w, self.rnn_input_size)
        # RNN
        x = self.rnn(x)
        return x, bn
