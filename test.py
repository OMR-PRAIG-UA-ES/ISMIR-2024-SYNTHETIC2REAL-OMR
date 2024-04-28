import gc
import os

import fire
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers.wandb import WandbLogger

from my_utils.seed import seed_everything
from my_utils.dataset import CTCDataModule
from networks.base.model import CTCTrainedCRNN
from networks.amd.da_model import DATrainedCRNN

seed_everything(42)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()


def test(
    train_ds_name: str,
    test_ds_name: str,
    checkpoint_path: str,
    num_workers: int = 10,
    project: str = "AMD-Synth2Real-OMR",
    group: str = "Baseline-LowerBound",
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
    print(f"\tTest dataset: {test_ds_name}")
    print(f"\tCheckpoint path ({train_ds_name}): {checkpoint_path}")

    # Model
    try:
        model = CTCTrainedCRNN.load_from_checkpoint(checkpoint_path)
    except Exception:
        try:
            model = DATrainedCRNN.load_from_checkpoint(checkpoint_path)
        except Exception:
            raise ValueError(
                f"Could not load model from checkpoint path {checkpoint_path}"
            )
    model.freeze()

    # Get datamodule
    datamodule = CTCDataModule(
        ds_name=test_ds_name,
        exp_type="test",
        encoding_type=model.encoding_type,
        use_train_data_augmentation=False,
        train_batch_size=1,
        num_workers=num_workers,
    )
    datamodule.setup("test")

    # Test
    run_name = (
        f"{model.encoding_type.upper()}-Train-{train_ds_name}_Test-{test_ds_name}"
    )
    trainer = Trainer(
        logger=WandbLogger(
            project=project,
            group=group,
            name=run_name,
            log_model=False,
            config=args,
            entity="grfia",
        ),
        precision="16-mixed",
    )
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    fire.Fire(test)
