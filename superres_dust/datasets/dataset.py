import torch
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import Dataset


class SRDDataset(Dataset):
    def __init__(self, data: torch.Tensor, resol: torch.Tensor):
        """
        Parameters:
            data: Tensor containing the data with indices
                corresponding to: tau, q_hi, u_hi, q_lr,
                u_lr, q_hr, u_hr
            resol: Tensor containing the angular resolutio
                of the output images.
        """
        self.data = data
        self.resol = resol

        self.dataset_size = len(self.data)
        rank_zero_info(f"\tDataset size: {self.dataset_size}")

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return self.dataset_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
            idx: Index of the sample to retrieve.
        Returns:
            Tuple containing the data and resolution of the
            sample at the given index.
        """
        return self.data[idx], self.resol[idx]
