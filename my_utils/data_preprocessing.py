import re
import joblib

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F

MEMORY = joblib.memory.Memory("./cache", mmap_mode="r", verbose=0)

NUM_CHANNELS = 1
IMG_HEIGHT = 64

################################# Image preprocessing:


@MEMORY.cache
def preprocess_image_from_file(path, eps=1e-7):
    # Convert to grayscale
    x = Image.open(path).convert("L")
    # Resize (preserving aspect ratio)
    new_width = int(IMG_HEIGHT * x.size[0] / x.size[1])
    x = x.resize((new_width, IMG_HEIGHT))
    # Convert to numpy array
    x = np.array(x, dtype=np.float32)
    # Normalize to [0, 1]
    x = (x - np.amin(x)) / (np.amax(x) - np.amin(x) + eps)
    # Add channel dimension
    x = np.expand_dims(x, axis=0)
    # Convert to tensor
    x = torch.from_numpy(x)
    return x


################################# Transcript preprocessing:


@MEMORY.cache
def preprocess_transcript_from_file(
    path,
    w2i,
    encoding_type="standard",
):
    with open(path, "r") as file:
        y = file.read().strip()
    if encoding_type == "standard":
        # Ex.: y = ["clef:G2", "note.black:L1" ...]
        y = y.split()
    else:
        # encoding_type == "split"
        # Ex.: y = ["clef", "G2, "note.black", "L1" ...]
        y = re.split(r"\s+|:", y)
    return torch.tensor([w2i[w] for w in y])


################################# CTC Preprocessing:


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(torch.int32)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl
