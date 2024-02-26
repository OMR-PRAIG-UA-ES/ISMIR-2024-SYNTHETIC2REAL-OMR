import os
import re
import json

import torch
from torch.utils.data import Dataset

from my_utils.data_preprocessing import (
    preprocess_image_from_file,
    preprocess_transcript_from_file,
)

################################################################################################ Single-source:


class CTCDataset(Dataset):
    def __init__(
        self,
        name,
        samples_filepath,
        transcripts_folder,
        img_folder,
        train=True,
        da_train=False,
        width_reduction=2,
        encoding_type="standard",  # Standard or split
    ):
        self.name = name
        self.train = train
        self.da_train = da_train
        self.width_reduction = width_reduction
        self.encoding_type = encoding_type

        # Remove the appendix _up and _down from the notes in FMT and Malaga as
        # Primus/CameraPrimus datasets do not have these appendixes
        self.remove_stem_direction = True if self.name in ["FMT", "Malaga"] else False

        # Get image paths and transcripts
        self.X, self.Y = self.get_images_and_transcripts_filepaths(
            samples_filepath, img_folder, transcripts_folder
        )

        # Check and retrieve vocabulary
        vocab_name = f"w2i_{self.encoding_type}.json"
        vocab_folder = os.path.join(os.path.join("data", self.name), "vocab")
        os.makedirs(vocab_folder, exist_ok=True)
        self.w2i_path = os.path.join(vocab_folder, vocab_name)
        self.w2i, self.i2w = self.check_and_retrieve_vocabulary(transcripts_folder)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.da_train:
            # Domain Adaptation setting
            x = preprocess_image_from_file(self.X[idx])
            return x

        else:
            # CTC Training setting
            x = preprocess_image_from_file(self.X[idx])
            y = preprocess_transcript_from_file(
                self.Y[idx],
                self.w2i,
                self.encoding_type,
                self.remove_stem_direction,
            )

            if self.train:
                # x.shape = [channels, height, width]
                return x, x.shape[2] // self.width_reduction, y, len(y)

            return x, y

    def get_images_and_transcripts_filepaths(
        self, img_dat_file_path, img_directory, transcripts_directory
    ):
        images = []
        transcripts = []

        # Images and transcripts are in different directories
        # Image filepath example: {image_name}.jpg
        # Transcript filepath example: {image_name}.jpg.txt

        # We are using the agnostic encoding for the transcripts

        # Read the .dat file to get the image file names
        with open(img_dat_file_path, "r") as file:
            img_files = file.read().splitlines()

        for img_file in img_files:
            img_path = os.path.join(img_directory, img_file)

            transcript_file = img_file + ".txt"
            transcript_path = os.path.join(transcripts_directory, transcript_file)

            if os.path.exists(img_path) and os.path.exists(transcript_path):
                images.append(img_path)
                transcripts.append(transcript_path)

        return images, transcripts

    def check_and_retrieve_vocabulary(self, transcripts_dir):
        w2i = {}
        i2w = {}

        if os.path.isfile(self.w2i_path):
            with open(self.w2i_path, "r") as file:
                w2i = json.load(file)
            i2w = {v: k for k, v in w2i.items()}
        else:
            w2i, i2w = self.make_vocabulary(transcripts_dir)
            with open(self.w2i_path, "w") as file:
                json.dump(w2i, file)

        return w2i, i2w

    def make_vocabulary(self, transcripts_dir):
        vocab = set()
        for transcript_file in os.listdir(transcripts_dir):
            with open(os.path.join(transcripts_dir, transcript_file), "r") as file:
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
        if self.remove_stem_direction:
            vocab = [w.replace("_up", "").replace("_down", "") for w in vocab]
            vocab = sorted(set(vocab))

        w2i = {}
        i2w = {}
        for i, w in enumerate(vocab):
            w2i[w] = i + 1
            i2w[i + 1] = w
        w2i["<PAD>"] = 0
        i2w[0] = "<PAD>"

        return w2i, i2w
