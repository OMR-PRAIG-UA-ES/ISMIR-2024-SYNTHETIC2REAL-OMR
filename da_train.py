import gc
import os

import fire
import torch

from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


from my_utils.seed import seed_everything
from my_utils.dataset import CTCDataModule
from networks.amd.da_model import DATrainedCRNN

seed_everything(42)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def da_train(
    # Datasets and model
    train_ds_name: str,
    test_ds_name: str,
    checkpoint_path: str,
    # Training hyperparameters
    bn_ids: list[int],
    align_loss_weight: float = 1.0,
    minimize_loss_weight: float = 1.0,
    diversify_loss_weight: float = 1.0,
    lr: float = 3e-4,
    encoding_type: str = "standard",
    epochs: int = 1000,
    patience: int = 20,
    train_batch_size: int = 16,
    num_workers: int = 10,
    # Callbacks
    project: str = "AMD-Synth2Real-OMR",
    group: str = "Source-Free-Adaptation",
    return_run_stats: bool = False,
):
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # Get all the inputs to the function as a dictionary
    args = dict(locals())

    # Check if checkpoint path exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    # Experiment info
    print(f"Running experiment: {project} - {group}")
    print(f"\tSource model ({train_ds_name}): {checkpoint_path}")
    print(f"\tTarget dataset: {test_ds_name}")
    print(f"\tEncoding type: {encoding_type}")
    print(f"\tBN identifiers: {bn_ids}")
    print("\tLoss weights factors:")
    print(f"\t\talign_loss_weight={align_loss_weight}")
    print(f"\t\tminimize_loss_weight={minimize_loss_weight}")
    print(f"\t\tdiversify_loss_weight={diversify_loss_weight}")
    print(f"\tLearning rate: {lr}")
    print(f"\tEpochs: {epochs}")
    print(f"\tPatience: {patience}")
    print(f"\tTrain Batch Size: {train_batch_size}")
    print(f"\tNum Workers: {num_workers}")

    # Get datamodule
    datamodule = CTCDataModule(
        ds_name=test_ds_name,
        exp_type="da_train",
        encoding_type=encoding_type,
        use_train_data_augmentation=False,
        train_batch_size=train_batch_size,
        num_workers=num_workers,
    )
    datamodule.setup("fit")

    # Model
    bn_ids = [bn_ids] if type(bn_ids) == int else bn_ids
    model = DATrainedCRNN(src_checkpoint_path=checkpoint_path, bn_ids=bn_ids)
    model_name = f"{encoding_type.upper()}-Train-{train_ds_name}_Test-{test_ds_name}"
    model_name += f"_lr{lr}_bn{'-'.join([str(bn_id) for bn_id in bn_ids])}"
    model_name += (
        f"_a{align_loss_weight}_m{minimize_loss_weight}_d{diversify_loss_weight}"
    )

    # Loss
    model.configure_da_loss(
        lr=lr,
        align_loss_weight=align_loss_weight,
        minimize_loss_weight=minimize_loss_weight,
        diversify_loss_weight=diversify_loss_weight,
    )

    # Train and validate
    callbacks = [
        ModelCheckpoint(
            dirpath=f"weights/{group}",
            filename=model_name,
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
            name=model_name,
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
    model = DATrainedCRNN.load_from_checkpoint(callbacks[0].best_model_path)
    model.freeze()
    trainer.test(model, datamodule=datamodule)

    # Remove checkpoint
    if return_run_stats:
        return {
            "checkpoint_path": callbacks[0].best_model_path,
            "test_ser": trainer.callback_metrics["test_ser"],
        }


if __name__ == "__main__":
    fire.Fire(da_train)
