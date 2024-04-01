import os
import gc

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from my_utils.seed import seed_everything
from my_utils.dataset import CTCDataModule
from networks.base.model import CTCTrainedCRNN

seed_everything(42, deterministic=False)  # CTC does not have deterministic mode

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def train(
    ds_name: str,
    encoding_type: str = "standard",
    use_train_data_augmentation: bool = True,
    epochs: int = 1000,
    patience: int = 20,
    train_batch_size: int = 16,
    num_workers: int = 10,
    project: str = "AMD-Synth2Real-OMR",
    group: str = "Baseline-UpperBound",
):
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Get all the inputs to the function as a dictionary
    args = dict(locals())

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tDataset: {ds_name}")
    print(f"\tEncoding type: {encoding_type}")
    print(f"\tTrain Augmentations?: {use_train_data_augmentation}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tTrain Batch Size: {train_batch_size}")
    print(f"\tNum Workers: {num_workers}")

    # Get datamodule
    datamodule = CTCDataModule(
        ds_name=ds_name,
        exp_type="train",
        encoding_type=encoding_type,
        use_train_data_augmentation=use_train_data_augmentation,
        train_batch_size=train_batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("fit")

    # Model
    model = CTCTrainedCRNN(
        w2i=datamodule.get_w2i(),
        i2w=datamodule.get_i2w(),
        encoding_type=encoding_type,
    )

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=f"{ds_name}_{encoding_type}",
            monitor="val_ser",
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
            monitor="val_ser",
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
            config=args,
            entity="grfia",
        ),
        callbacks=callbacks,
        max_epochs=epochs,
        check_val_every_n_epoch=1,
        precision="16-mixed",
    )
    trainer.fit(model, datamodule=datamodule)

    # Test
    model = CTCTrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(train)
