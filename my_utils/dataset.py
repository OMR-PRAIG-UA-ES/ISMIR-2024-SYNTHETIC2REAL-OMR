import os
import re
import json

import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from data.config import DATASETS, DS_CONFIG
from my_utils.augmentations import AugmentStage
from my_utils.data_preprocessing import (
    ENCODING_TYPES,
    preprocess_image_from_file,
    preprocess_transcript_from_file,
    ctc_batch_preparation,
    pad_batch_images,
)

EXPERIMENT_TYPES = ["train", "test", "da_train"]


class CTCDataModule(LightningDataModule):
    def __init__(
        self,
        ds_name: str,
        exp_type: str = "train",
        encoding_type: str = "standard",
        use_train_data_augmentation: bool = True,
        train_batch_size: int = 16,
        num_workers: int = 10,
    ):
        super(CTCDataModule, self).__init__()
        # Dataset folder
        self.ds_name = ds_name
        self.x_folder = DS_CONFIG[self.ds_name]["images"]
        self.y_folder = DS_CONFIG[self.ds_name]["transcripts"]
        # Data augmentation/preprocessing
        self.exp_type = exp_type
        self.encoding_type = encoding_type
        self.use_train_data_augmentation = use_train_data_augmentation
        self.train_batch_size = train_batch_size
        self.collate_fn = (
            pad_batch_images if exp_type == "da_train" else ctc_batch_preparation
        )
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_ds = CTCDataset(
                ds_name=self.ds_name,
                x_folder=self.x_folder,
                y_folder=self.y_folder,
                partition_filepath=DS_CONFIG[self.ds_name]["train"],
                use_train_data_augmentation=self.use_train_data_augmentation,
                exp_type=self.exp_type,
                encoding_type=self.encoding_type,
            )
            self.valid_ds = CTCDataset(
                ds_name=self.ds_name,
                x_folder=self.x_folder,
                y_folder=self.y_folder,
                partition_filepath=DS_CONFIG[self.ds_name]["val"],
                use_train_data_augmentation=False,
                exp_type="test",
                encoding_type="split",  # Always split for evaluation (fair comparison between encodings)
            )

        if stage == "test" or stage == "predict":
            self.test_ds = CTCDataset(
                ds_name=self.ds_name,
                x_folder=self.x_folder,
                y_folder=self.y_folder,
                partition_filepath=DS_CONFIG[self.ds_name]["test"],
                use_train_data_augmentation=False,
                exp_type="test",
                encoding_type="split",  # Always split for evaluation (fair comparison between encodings)
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )  # prefetch_factor=2

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )  # prefetch_factor=2

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader(self)

    def get_w2i(self) -> dict[str, int]:
        try:
            return self.train_ds.w2i
        except AttributeError:
            return self.test_ds.w2i

    def get_i2w(self) -> dict[int, str]:
        try:
            return self.train_ds.i2w
        except AttributeError:
            return self.test_ds.i2w


##################################################################################


class CTCDataset(Dataset):
    def __init__(
        self,
        ds_name: str,
        x_folder: str,
        y_folder: str,
        partition_filepath: str,
        use_train_data_augmentation: bool = True,
        exp_type: str = "train",
        encoding_type: str = "standard",
    ):
        # Check if dataset exists
        if ds_name not in DATASETS:
            raise NotImplementedError(f"Dataset {ds_name} not implemented")
        self.ds_name = ds_name

        # Check if experiment type is valid
        if exp_type not in EXPERIMENT_TYPES:
            raise ValueError(f"Invalid experiment type: {exp_type}")
        self.exp_type = exp_type

        # Check if encoding type is valid
        if encoding_type not in ENCODING_TYPES:
            raise ValueError(f"Invalid encoding type: {encoding_type}")
        self.encoding_type = encoding_type

        # Set image and transcript folders
        self.x_folder = x_folder
        self.y_folder = y_folder
        self.partition_filepath = partition_filepath

        # Get image and transcript filepaths
        self.XFiles, self.YFiles = self.get_images_and_transcripts_filepaths(
            partition_filepath=partition_filepath
        )

        # Check and retrieve vocabulary
        vocab_name = f"w2i_{self.encoding_type}.json"
        vocab_folder = os.path.join(os.path.join("data", self.ds_name), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary()

        # Set data augmentation
        self.augment = AugmentStage() if use_train_data_augmentation else lambda x: x

        # Set the __getitem__ method
        self.__setgetitem__()

    def get_images_and_transcripts_filepaths(
        self, partition_filepath: str
    ) -> tuple[list[str], list[str]]:
        images = []
        transcripts = []

        # Read the .dat file to get the image file names
        with open(partition_filepath, "r") as file:
            img_files = file.read().splitlines()

        for img_file in img_files:
            # Images and transcripts are in different directories
            # Image filepath example: {image_name}.jpg
            # Transcript filepath example: {image_name}.jpg.txt

            # Image
            img_path = os.path.join(self.x_folder, img_file)
            # Transcript
            transcript_file = img_file + ".txt"
            transcript_path = os.path.join(self.y_folder, transcript_file)

            if os.path.exists(img_path) and os.path.exists(transcript_path):
                images.append(img_path)
                transcripts.append(transcript_path)

        return images, transcripts

    ###################################################### Vocabulary:

    def check_and_retrieve_vocabulary(self) -> tuple[dict[str, int], dict[int, str]]:
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary()
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self) -> tuple[dict[str, int], dict[int, str]]:
        vocab = set()
        for transcript_file in os.listdir(self.y_folder):
            if not transcript_file.endswith(".txt") or transcript_file.startswith("."):
                continue
            with open(os.path.join(self.y_folder, transcript_file), "r") as file:
                if self.encoding_type == "standard":
                    # Ex.: y = ["clef:G2", "note.black:L1" ...]
                    words = file.read().strip().split()
                else:
                    # encoding_type == "split"
                    # Split each transcript into words/tokens using spaces and ':'
                    # Ex.: y = ["clef", "G2", "note.black", "L1" ...]
                    words = re.split(r"\s+|:", file.read().strip())
                vocab.update(words)
        vocab = sorted(vocab)

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w

    ###################################################### Getters:

    def __len__(self) -> int:
        return len(self.XFiles)

    def __setgetitem__(self) -> callable:
        if self.exp_type == "train":
            self.__getitem_func = self.__traingetitem__
        elif self.exp_type == "test":
            self.__getitem_func = self.__validgetitem__
        elif self.exp_type == "da_train":
            self.__getitem_func = self.__datraingetitem__
        else:
            raise ValueError(f"Invalid experiment type: {self.exp_type}")

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, int, torch.Tensor, int] | tuple[torch.Tensor, list[str]]:
        return self.__getitem_func(idx)

    def __traingetitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor, int]:
        # Image
        x = preprocess_image_from_file(path=self.XFiles[idx])
        x = self.augment(x)
        # Transcript
        y = preprocess_transcript_from_file(
            path=self.YFiles[idx],
            encoding_type=self.encoding_type,
        )
        y = torch.tensor([self.w2i[w] for w in y], dtype=torch.int32)
        return x, x.shape[2], y, len(y)

    def __validgetitem__(self, idx: int) -> tuple[torch.Tensor, list[str]]:
        # Inference is performed on a per-sample basis
        # Image
        x = preprocess_image_from_file(path=self.XFiles[idx])
        # Transcript (return the transcript as a list of strings)
        y = preprocess_transcript_from_file(
            path=self.YFiles[idx],
            encoding_type=self.encoding_type,
        )
        return x, y

    def __datraingetitem__(self, idx: int) -> torch.Tensor:
        # Image
        x = preprocess_image_from_file(path=self.XFiles[idx])
        return x
