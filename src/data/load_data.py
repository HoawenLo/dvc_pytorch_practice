import yaml

from torch import stack, Tensor
from torchvision import datasets, transforms

from src.logging.log import get_logger

def load_data(config):
    """"Loads the CIFAR100 data.

    Args:
        config: Configuration file. The params.yaml file.

    Returns:
        Training images dataset and labels. Validation images dataset and labels. 
    """

    logger = get_logger("Load data")

    root = config["load_data"]["root"]

    logger.info(f"Importing training and validation dataset CIFAR100 from {root}")
    train_data = datasets.CIFAR100(root=config["load_data"]["root"], train=config["load_data"]["train"], transform=transforms.ToTensor())
    val_data = datasets.CIFAR100(root=config["load_data"]["root"], train=config["load_data"]["val"], transform=transforms.ToTensor())
    logger.info(f"Data loaded.")

    logger.info(f"Extract images from dataset.")
    train_imgs = stack([img[0] for img in train_data])
    val_imgs = stack([img[0] for img in val_data])
    logger.info(
        f"Images extracted from training and validation dataset. "
        f"Training image dataset has shape: {train_imgs.shape} "
        f"Validation image dataset has shape: {val_imgs.shape}"
    )

    logger.info(f"Extract labels from dataset.")
    train_labels = Tensor([label[1] for label in train_data])
    val_labels = Tensor([label[1] for label in val_data])
    logger.info(
        f"Labels extracted from training and validation dataset. "
        f"Training label dataset has shape: {train_labels.shape} "
        f"Validation label dataset has shape: {val_labels.shape}"
    )

    return train_imgs, train_labels, val_imgs, val_labels