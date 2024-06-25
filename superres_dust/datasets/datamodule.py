import pickle
from pathlib import Path
import os

import numpy as np
from lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import random_split, DataLoader, Dataset

from datasets import SRDDataset
from transforms import Normalize, Augment


class SRDDataModule(LightningDataModule):
    """Data module for data preprocessing and loading.

    Methods:
        prepare_data: Preprocesses data and splits it into train, val, and test sets.
        train_dataloader: Returns a DataLoader for the training set.
        val_dataloader: Returns a DataLoader for the validation set.
        test_dataloader: Returns a DataLoader for the test set.
        predict_dataloader: Returns a DataLoader for the prediction set.
        get_dataloader: Returns a DataLoader for a given dataset.
    """

    def __init__(self, config: dict):
        """
        Parameters:
            config: Dictionary containing configuration parameters.
        """
        super().__init__()

        self.num_workers = 0 if config["debug"] else config["num_workers"]
        self.pin_memory = not config["debug"]
        self.persistent_workers = not config["debug"]
        self.batch_size = config["batch_size"]
        self.data_dir = Path(config["dir"])

        data_boundaries = pickle.load(open(self.data_dir / "data_boundaries.pkl", "rb"))
        self.normalize = Normalize(torch.from_numpy(data_boundaries))

    def prepare_data(self):
        """Preprocess data and split it into train, val, and test sets.
        If preprocessed data exists, it is loaded in.
        """
        # Check if preprocessed data exists
        if (
            os.path.exists(self.data_dir / "train_subset.pt")
            and os.path.exists(self.data_dir / "val_subset.pt")
            and os.path.exists(self.data_dir / "test_subset.pt")
        ):
            rank_zero_info("Loading preprocessed data splits...")
            self.train_subset = torch.load(self.data_dir / "train_subset.pt")
            self.val_subset = torch.load(self.data_dir / "val_subset.pt")
            self.test_subset = torch.load(self.data_dir / "test_subset.pt")

        else:
            rank_zero_info("Preprocessing data...")
            data = []
            resols = []

            # Load pickled samples and resol
            for filename in os.listdir(self.data_dir):
                if filename.startswith("resol"):
                    with open(self.data_dir / filename, "rb") as f:
                        sample = pickle.load(f)
                        # Extract resolution from filename
                        resol = int(filename.split("resol")[1][:2])
                        data.append(sample)
                        resols.append(resol)

            # tau, q_hi, u_hi, q_lr, u_lr, q_hr, u_hr
            data = torch.tensor(np.array(data)).unsqueeze(2)
            resols = torch.tensor(np.array(resols), dtype=torch.float32)

            # Normalize data
            normalized_data = self.normalize(data)  # .half()

            # Split data into train, val, and test sets
            rank_zero_info("\tPre-augmentation:")
            dataset = SRDDataset(normalized_data, resols)
            train_subset, val_subset, test_subset = random_split(
                dataset, [0.8, 0.1, 0.1]
            )

            # Perform augmentations on all splits
            augmented_datasets = []
            for dataset, split_name in zip(
                [train_subset, val_subset, test_subset], ["train", "val", "test"]
            ):
                augmented_data = []
                augmented_resol = []
                for sample, resol in dataset:
                    aug = Augment(sample)
                    augmented_samples = aug.get_all()
                    augmented_data.extend(augmented_samples)
                    augmented_resol.extend([resol] * 8)

                augmented_data = torch.stack(augmented_data).float()
                augmented_resol = torch.tensor(augmented_resol, dtype=torch.float32)
                rank_zero_info(f"\tPost-augmentation {split_name}:")
                augmented_datasets.append(SRDDataset(augmented_data, augmented_resol))

            self.train_subset, self.val_subset, self.test_subset = augmented_datasets

            # Save preprocessed datasets
            torch.save(self.train_subset, self.data_dir / "train_subset.pt")
            torch.save(self.val_subset, self.data_dir / "val_subset.pt")
            torch.save(self.test_subset, self.data_dir / "test_subset.pt")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self.get_dataloader(self.train_subset, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.val_subset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.test_subset)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return self.get_dataloader(self.test_subset)

    def get_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Returns a DataLoader for a given dataset.

        Parameters:
            dataset: Dataset to load.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader for the given dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
