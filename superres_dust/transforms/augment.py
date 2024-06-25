from typing import Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as F


class Augment:
    """Augment an image by rotating and flipping it."""

    def __init__(self, imgs: torch.Tensor):
        self.imgs = imgs

    def rotate(self, k: int) -> torch.Tensor:
        """Rotate the image k times.

        Parameters:
            k (int): Number of times to rotate the image.

        Returns:
            torch.Tensor: Rotated image.
        """
        return torch.rot90(self.imgs, k=k, dims=[-2, -1])

    def flip(self, k: Optional[int] = None) -> torch.Tensor:
        """Flip the image horizontally. If k is not None,
        rotate the flipped image k times.

        Parameters:
            k (Optional[int]): Number of times to
            rotate the flipped image. Defaults to None.

        Returns:
            torch.Tensor: Flipped image (optionally rotated).
        """
        if k is None:
            return F.hflip(self.imgs)
        return F.hflip(self.rotate(k))

    def get_all(self) -> list[torch.Tensor]:
        """Get all augmented images.

        Returns:
            list[torch.Tensor]: List of augmented images.
        """
        aug_imgs = [self.imgs, self.flip()]
        for k in range(1, 4):
            aug_imgs.append(self.rotate(k))
            aug_imgs.append(self.flip(k))
        return aug_imgs
