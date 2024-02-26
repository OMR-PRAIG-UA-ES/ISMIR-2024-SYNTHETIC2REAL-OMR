import gc
import os
import random

import fire
import torch
import numpy as np
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger

from data.config import DS_CONFIG
from my_utils.dataset import CTCDataset
from networks.base.model import CTCTrainedCRNN

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def test(
    train_ds_name,
    test_ds_name,
    checkpoint_path,
    encoding_type="standard",
    project="AMD-Self-Labelled-OMR",
    group="Baseline-LowerBound",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Check if datasets exist
    if train_ds_name not in DS_CONFIG.keys():
        raise NotImplementedError(f"Train dataset {train_ds_name} not implemented")
    if test_ds_name not in DS_CONFIG.keys():
        raise NotImplementedError(f"Test dataset {test_ds_name} not implemented")

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tTest dataset: {test_ds_name}")
    print(f"\tCheckpoint path ({train_ds_name}): {checkpoint_path}")
    print(f"\tEncoding type: {encoding_type}")

    # Dataset
    test_ds = CTCDataset(
        name=test_ds_name,
        samples_filepath=DS_CONFIG[test_ds_name]["test"],
        transcripts_folder=DS_CONFIG[test_ds_name]["transcripts"],
        img_folder=DS_CONFIG[test_ds_name]["images"],
        train=False,
        encoding_type=encoding_type,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=20
    )  # prefetch_factor=2

    # Model
    model = CTCTrainedCRNN.load_from_checkpoint(checkpoint_path, ytest_i2w=test_ds.i2w)
    model.freeze()

    # Test: automatically auto-loads the best weights from the previous run
    run_name = f"{encoding_type.upper()}-Train-{train_ds_name}_Test-{test_ds_name}"
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=run_name,
            log_model=False,
            entity="grfia",
        ),
        precision="16-mixed",
    )
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(test)
