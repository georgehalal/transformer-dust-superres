import torch


class Normalize:
    """Normalize the input images."""

    def __init__(
        self,
        boundaries: torch.Tensor,
    ):
        """Initialize the Normalize class.

        Parameters:
            boundaries (torch.Tensor): Boundaries for normalization.
        """
        self.boundaries = [boundaries[0]] + [boundaries[1]] * 2 + [boundaries[2]] * 4

    def normalize_image(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        """Normalize the image to the range [0, 1] or [-1, 1].

        Parameters:
            image (torch.Tensor): Image to normalize.
            idx (int): Index of the image in the input tensor.

        Returns:
            torch.Tensor: Normalized image.
        """

        # Clip the image after normalization to prevent rounding errors
        image_max = self.boundaries[idx].item()
        if idx == 0:
            image_min = 0.0
            image = torch.clamp(image, min=image_min, max=image_max)
            image = (image - image_min) / (image_max - image_min)
            image = torch.clamp(image, min=0.0, max=1.0)
        else:
            image_min = -image_max
            image = torch.clamp(image, min=image_min, max=image_max)
            image = 2 * (image - image_min) / (image_max - image_min) - 1
            image = torch.clamp(image, min=-1.0, max=1.0)

        return image

    def denormalize_image(self, image: torch.Tensor, idx: int) -> torch.Tensor:
        """Denormalize the image from the range [0, 1] or [-1, 1].

        Parameters:
            image (torch.Tensor): Image to denormalize.
            idx (int): Index of the image in the input tensor.

        Returns:
            torch.Tensor: Denormalized image.
        """
        # Clip the image after denormalization to prevent rounding errors
        image_max = self.boundaries[idx].float().item()
        if idx == 0:
            image_min = 0.0
            image = image * (image_max - image_min) + image_min
        else:
            image_min = -image_max
            image = 0.5 * (image + 1) * (image_max - image_min) + image_min
        image = torch.clamp(image, min=image_min, max=image_max)

        return image

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize each input image separately and stack them.

        Parameters:
            images (torch.Tensor): Input tensor to normalize.
                Must have size N x 7 x 80 x 80, where N is the
                batch size. The 7 channels are tau, q_hi, u_hi,
                q_lr, u_lr, q_hr, and u_hr in this order.

        Returns:
            torch.Tensor: Normalized input tensor.
        """
        if images.dim() == 5:
            assert images.size(1) == 7
            return torch.stack(
                [self.normalize_image(images[:, i, ...], i) for i in range(7)],
                dim=1
            )
        else:
            raise NotImplementedError(
                "Normalize call not implemented for a single map"
            )
