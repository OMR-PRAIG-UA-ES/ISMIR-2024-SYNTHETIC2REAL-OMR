import os
import shutil
import tarfile
import requests
from typing import List

import numpy as np
from sklearn.model_selection import KFold, train_test_split


def download_and_extract_camera_primus_dataset():
    file_path = "CameraPrIMuS.tgz"
    extract_path = "."

    # Download dataset
    response = requests.get(
        url="https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz"
    )
    with open(file_path, "wb") as file:
        file.write(response.content)
    # Extract dataset
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(extract_path)
    # Remove tar file
    os.remove(file_path)


def create_kfolds(samples: List[str], folds_dir: str, k: int = 5):
    kf = KFold(n_splits=k, random_state=42, shuffle=True)

    samples = np.array(samples)

    i = 0
    for train_index, test_index in kf.split(samples):
        train_fold = os.path.join(folds_dir, f"train_gt_fold{i}.dat")
        val_fold = os.path.join(folds_dir, f"val_gt_fold{i}.dat")
        test_fold = os.path.join(folds_dir, f"test_gt_fold{i}.dat")

        X_train, X_test = samples[train_index], samples[test_index]
        X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

        with open(train_fold, "w") as txt:
            for img in X_train:
                txt.write(img + "\n")
        with open(val_fold, "w") as txt:
            for img in X_val:
                txt.write(img + "\n")
        with open(test_fold, "w") as txt:
            for img in X_test:
                txt.write(img + "\n")
        i += 1


def format_camera_primus_dataset():
    # Create directories
    os.makedirs("data/Primus/Images", exist_ok=True)
    os.makedirs("data/Primus/GT", exist_ok=True)
    os.makedirs("data/Primus/Folds", exist_ok=True)
    os.makedirs("data/CameraPrimus/Images", exist_ok=True)

    # Move images and transcripts to their corresponding directories
    for sample_dir in os.listdir("Corpus"):
        if sample_dir.startswith("."):
            continue
        for f in os.listdir(os.path.join("Corpus", sample_dir)):
            if f.startswith("."):
                continue
            if f.endswith(".png"):
                # Move image to data/Primus/Images
                # and change file extension to .jpg
                os.rename(
                    os.path.join("Corpus", sample_dir, f),
                    os.path.join("data/Primus/Images", f.replace(".png", ".jpg")),
                )
            elif f.endswith(".agnostic"):
                # Move transcript to data/Primus/GT
                # and change file extension to .jpg.txt
                os.rename(
                    os.path.join("Corpus", sample_dir, f),
                    os.path.join("data/Primus/GT", f.replace(".agnostic", ".jpg.txt")),
                )
            elif f.endswith("_distorted.jpg"):
                # Move distorted image to data/CameraPrimus/Images
                os.rename(
                    os.path.join("Corpus", sample_dir, f),
                    os.path.join("data/CameraPrimus/Images", f.replace("_distorted.jpg", ".jpg")),
                )

    # Create folds
    if len(os.listdir("data/Primus/Folds")) == 0:
        create_kfolds(
            samples=[
                f
                for f in os.listdir("data/Primus/Images")
                if f.endswith(".jpg") and not f.startswith(".")
            ],
            folds_dir="data/Primus/Folds",
        )
    else:
        print("Folds already exist")

    # Delete Corpus dir and its contents
    shutil.rmtree("Corpus")

def change_separator():
    # Music symbols consists of two terms: event durationg and pitch
    # In the original dataset, the separator between these two terms is a "-"
    # Changed to ":" to be consistent with the rest of datasets
    # If "-" is used in after "L" or "S" (and followed by a number), it is not changed
    for f in os.listdir("data/Primus/GT"):
        if f.startswith("."):
            continue
        with open(os.path.join("data/Primus/GT", f), "r") as txt:
            content = txt.read()
        content = content.replace("L-", "L:")
        content = content.replace("S-", "S:")
        content = content.replace("-", ":")
        content = content.replace("L:", "L-")
        content = content.replace("S:", "S-")
        with open(os.path.join("data/Primus/GT", f), "w") as txt:
            txt.write(content)

def create_camera_primus_dataset():
    # Check if Corpus dir exists
    if not os.path.exists("Corpus"):
        # Download and extract dataset
        download_and_extract_camera_primus_dataset()
    # Format dataset
    format_camera_primus_dataset()
    # Change separator
    change_separator()

if __name__ == "__main__":
    create_camera_primus_dataset()
