import os
import pandas as pd
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torchvision.datasets import ImageFolder


class ImageFolderWithText(ImageFolder):
    """
    Custom dataset that extends ImageFolder to include text captions.

    Args:
        root: Root directory path of the image dataset
        captions_csv_path: Path to the CSV file containing image captions
        transform: Optional transform to be applied on images
        target_transform: Optional transform to be applied on labels
    """

    def __init__(
        self,
        root: str,
        captions_csv_path: str,
        transform: Optional[Callable[[Any], Any]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)

        # Load captions from CSV
        self.captions_df = pd.read_csv(captions_csv_path)

        # Create a mapping from image filename to caption
        self.image_to_caption = {}
        for _, row in self.captions_df.iterrows():
            image_filename = row['image_path']
            caption = row['caption']
            self.image_to_caption[image_filename] = caption

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index: Index

        Returns:
            tuple: (image, target, text) where target is the class index and text is the caption
        """
        # Get the original image and target from ImageFolder
        image, target = super().__getitem__(index)

        # Get the image path to extract filename
        path, _ = self.samples[index]
        image_filename = os.path.basename(path)

        # Get the corresponding caption
        caption = self.image_to_caption.get(image_filename, "")

        return image, target, caption
