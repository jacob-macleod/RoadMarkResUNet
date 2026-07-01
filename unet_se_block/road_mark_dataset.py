"""
Class to import the road mark dataset
"""

import os
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from unet_se_block.model_config import MODEL_CONFIG


class RoadMarkDataset(torch.utils.data.Dataset):
    """
    Class to import the road mark dataset
    """

    def __init__(self, data_dir, subfolder="train", transform=None):
        """
        Initialise the class

        Args:
            data_dir (str): The folder storing the data
            subfolder (str, optional): The subfolder within the main fodler. Defaults to "train".
            transform (callable, optional): Optional transform which can be applied.
                Defaults to None.
        """
        self.data_dir = data_dir
        self.subfolder = subfolder
        self.img_dir = os.path.join(self.data_dir, self.subfolder)

        self.annotations_file = os.path.join(self.img_dir, "_annotations.coco.json")
        self.coco = COCO(self.annotations_file)
        self.image_ids = sorted(self.coco.getImgIds())
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        """
        Used when getting a specific image
        """
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.img_dir, image_info["file_name"])
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate the binary mask
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        # Create an empty mask
        mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
        # Convert the polygon annotations to a binary mask
        for annotation in annotations:
            pixel_value = 1
            mask = np.maximum(mask, self.coco.annToMask(annotation) * pixel_value)

        image = cv2.resize(image, MODEL_CONFIG["img_size"])
        mask = cv2.resize(
            mask, MODEL_CONFIG["img_size"], interpolation=cv2.INTER_NEAREST
        )

        # Normalise it and convert it to a vector
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0).astype(np.float32)

        return torch.tensor(image), torch.tensor(mask)
