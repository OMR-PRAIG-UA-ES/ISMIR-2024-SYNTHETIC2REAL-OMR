import os
import gc
import random

import fire
import torch
import numpy as np
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger

from data.config import DS_CONFIG
from my_utils.dataset import CTCDataset
from networks.base.model import CTCTrainedCRNN
from my_utils.data_preprocessing import ctc_batch_preparation

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["WANDB_API_KEY"] = open("wandb_api_key.txt", "r").read().strip().close()


def train(
    ds_name,
    encoding_type="standard",
    epochs=1000,
    patience=20,
    batch_size=16,
    use_augmentations=True,
    metric_to_monitor="val_ser",
    project="AMD-Self-Labelled-OMR",
    group="Baseline-UpperBound",
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if dataset exists
    if ds_name not in DS_CONFIG.keys():
        raise NotImplementedError(f"Dataset {ds_name} not implemented")

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tDataset(s): {ds_name}")
    print(f"\tEncoding type: {encoding_type}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    # Get datasets
    train_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["train"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        encoding_type=encoding_type,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=ctc_batch_preparation,
    )  # prefetch_factor=2
    val_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["val"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        train=False,
        encoding_type=encoding_type,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4
    )  # prefetch_factor=2
    test_ds = CTCDataset(
        name=ds_name,
        samples_filepath=DS_CONFIG[ds_name]["test"],
        transcripts_folder=DS_CONFIG[ds_name]["transcripts"],
        img_folder=DS_CONFIG[ds_name]["images"],
        train=False,
        encoding_type=encoding_type,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4
    )  # prefetch_factor=2

    # Model
    model = CTCTrainedCRNN(
        w2i=train_ds.w2i, i2w=train_ds.i2w, use_augmentations=use_augmentations
    )
    train_ds.width_reduction = model.model.cnn.width_reduction

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=f"{ds_name}_{encoding_type}",
            monitor=metric_to_monitor,
            verbose=True,
            save_last=False,
            save_top_k=1,
            save_weights_only=False,
            mode="min",
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=metric_to_monitor,
            min_delta=0.1,
            patience=patience,
            verbose=True,
            mode="min",
            strict=True,
            check_finite=True,
            divergence_threshold=100.00,
            check_on_train_epoch_end=False,
        ),
    ]
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=f"{encoding_type.upper()}-Train-{ds_name}_Test-{ds_name}",
            log_model=False,
            entity="grfia",
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        deterministic=False,  # If True, raises error saying that CTC loss does not have this behaviour
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
        fast_dev_run=False,  # Set to True to check if everything is working
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    model = CTCTrainedCRNN.load_from_checkpoint(
        callbacks[0].best_model_path, ytest_i2w=test_ds.i2w
    )
    model.freeze()
    trainer.test(model, dataloaders=test_loader)


if __name__ == "__main__":
    fire.Fire(train)
