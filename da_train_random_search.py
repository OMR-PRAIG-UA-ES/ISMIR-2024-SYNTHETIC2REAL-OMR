import os
import random
import shutil
from itertools import combinations

import fire
import wandb

from da_train import da_train
from networks.base.modules import BN_IDS
from my_utils.seed import seed_everything

seed_everything(42)

# Set WANDB_API_KEY
with open("wandb_api_key.txt", "r") as f:
    os.environ["WANDB_API_KEY"] = f.read().strip()

# Sweep hyperparameters
WEIGHT_FACTORS = [1, 5, 10, 25, 50]
LRS = [1e-3, 3e-4]
ALL_BN_IDS = [
    combo for r in range(1, len(BN_IDS) + 1) for combo in combinations(BN_IDS, r)
]


def get_group_runs(
    project_name: str = "AMD-Synth2Real-OMR",
    group_name: str = "STANDARD-SZaragoza-TILS",
):
    api = wandb.Api()
    runs = api.runs(f"grfia/{project_name}", {"group": group_name})
    # Filter those runs that have finished
    runs = [run for run in runs if run.state == "finished"]
    names = [run.name for run in runs]
    return names


def run_sweep(
    train_ds_name: str,
    test_ds_name: str,
    encoding_type: str = "standard",
    num_random_combinations: int = 50,
):
    # Get runs
    group_name = f"{encoding_type.upper()}-S{train_ds_name}-T{test_ds_name}"
    names = get_group_runs(group_name=group_name)

    # Source checkpoint path (see train.py for details)
    checkpoint_path = (
        f"weights/Baseline-UpperBound/{train_ds_name}_{encoding_type}.ckpt"
    )

    # Log best SER values
    best_ser = float("inf")

    # Generate random hyperparameter combinations
    for _ in range(num_random_combinations):

        # Weight factors
        aw = random.choice(WEIGHT_FACTORS)
        mw = random.choice(WEIGHT_FACTORS)
        dw = random.choice(WEIGHT_FACTORS)

        # BN identifiers
        bn_ids = random.choice(ALL_BN_IDS)

        # Learning rate
        lr = random.choice(LRS)

        # Check if run exists
        run_name = f"{encoding_type.upper()}-Train-{train_ds_name}_Test-{test_ds_name}"
        run_name += f"_lr{lr}_bn{'-'.join(map(str, bn_ids))}_a{aw}_m{mw}_d{dw}"
        if run_name in names:
            continue

        # Run experiment
        run_stats = da_train(
            # Datasets and model
            train_ds_name=train_ds_name,
            test_ds_name=test_ds_name,
            checkpoint_path=checkpoint_path,
            # Training hyperparameters
            bn_ids=bn_ids,
            align_loss_weight=aw,
            minimize_loss_weight=mw,
            diversify_loss_weight=dw,
            lr=lr,
            encoding_type=encoding_type,
            # Callbacks
            group=group_name,
            return_run_stats=True,
        )
        # Log best SER
        if run_stats["test_ser"] < best_ser:
            best_ser = run_stats["test_ser"]
        else:
            # Remove checkpoint
            os.remove(run_stats["checkpoint_path"])
        # Remove wandb run directory
        run_dir = wandb.run.dir
        run_dir = run_dir.split("files")[0]
        wandb.finish()
        shutil.rmtree(run_dir)
        print("----------------------------------------")
    pass


if __name__ == "__main__":
    fire.Fire(run_sweep)
