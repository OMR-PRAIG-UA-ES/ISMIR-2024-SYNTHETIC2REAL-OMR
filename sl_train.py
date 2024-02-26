import gc
import os
import random

import fire
import torch
import numpy as np

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# Seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Deterministic behavior
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from data.config import DS_CONFIG
from my_utils.dataset import CTCDataset
from networks.self_labelling.model import SLTrainedCRNN
from my_utils.data_preprocessing import pad_batch_images


def self_labelled_train(
    # Datasets and model
    train_ds_name,
    test_ds_name,
    checkpoint_path,
    # Training hyperparameters
    confidence_threshold=0.9,
    encoding_type="standard",
    epochs=1000,
    patience=20,
    batch_size=16,
    use_augmentations=True,
    # Callbacks
    metric_to_monitor="val_ser",
    project="AMD-Self-Labelled-OMR",
    group="Self-Labelled-Adaptation",
    delete_checkpoint=False,
):
    gc.collect()
    torch.cuda.empty_cache()

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Check if datasets exist
    if not train_ds_name in DS_CONFIG.keys():
        raise NotImplementedError(f"Train dataset {train_ds_name} not implemented")
    if not test_ds_name in DS_CONFIG.keys():
        raise NotImplementedError(f"Test dataset {test_ds_name} not implemented")

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tSource model ({train_ds_name}): {checkpoint_path}")
    print(f"\tTarget dataset: {test_ds_name}")
    print(f"\tEncoding type: {encoding_type}")
    print(f"\tAugmentations: {use_augmentations}")
    print(f"\tConfidence threshold: {confidence_threshold}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tMetric to monitor: {metric_to_monitor}")

    # Get dataset
    train_ds = CTCDataset(
        name=test_ds_name,
        samples_filepath=DS_CONFIG[test_ds_name]["train"],
        transcripts_folder=DS_CONFIG[test_ds_name]["transcripts"],
        img_folder=DS_CONFIG[test_ds_name]["images"],
        train=True,
        encoding_type=encoding_type,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_batch_images,
    )  # prefetch_factor=2
    val_ds = CTCDataset(
        name=test_ds_name,
        samples_filepath=DS_CONFIG[test_ds_name]["val"],
        transcripts_folder=DS_CONFIG[test_ds_name]["transcripts"],
        img_folder=DS_CONFIG[test_ds_name]["images"],
        train=False,
        encoding_type=encoding_type,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=4
    )  # prefetch_factor=2
    test_ds = CTCDataset(
        name=test_ds_name,
        samples_filepath=DS_CONFIG[test_ds_name]["test"],
        transcripts_folder=DS_CONFIG[test_ds_name]["transcripts"],
        img_folder=DS_CONFIG[test_ds_name]["images"],
        train=False,
        encoding_type=encoding_type,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4
    )  # prefetch_factor=2

    # Model
    model = SLTrainedCRNN(
        train_dataloader=train_loader,
        src_checkpoint_path=checkpoint_path,
        ytest_i2w=train_ds.i2w,
        confidence_threshold=confidence_threshold,
        use_augmentations=use_augmentations,
    )
    model_name = f"{encoding_type.upper()}-Train-{train_ds_name}_Test-{test_ds_name}"
    model_name += f"_Confidence-{confidence_threshold}"
    model_name += f"_Augment-{use_augmentations}"

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=model_name,
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
            name=model_name,
            log_model=False,
            entity="grfia",
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        deterministic=True,
        benchmark=False,
        precision="16-mixed",  # Mixed precision training
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    model = SLTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, dataloaders=test_loader)

    # Remove checkpoint
    if delete_checkpoint:
        os.remove(callbacks[0].best_model_path)


if __name__ == "__main__":
    fire.Fire(self_labelled_train)
