import yaml

from torch import mean, std, long
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.logging.log import get_logger

# --------------------- Transformation --------------------- #

def calc_mean_std(input_data):
    """Calculate the mean and standard deviation across each channel.
    
    Args:
        input_data: The input data. Is a torch tensor.
        
    Returns:
        The mean and standard deviation of each colour channel of the dataset.
        The values are torch tensors of shape (3,)."""
    
    batch_size = input_data.shape[0]
    mean_val = mean(input_data.view(batch_size, 3, -1), dim=(0, 2))
    std_val = std(input_data.view(batch_size, 3, -1), dim=(0, 2))
    
    return mean_val, std_val

def apply_transformation(input_data):
    """Apply transformations to the data.
    
    Args:
        input_data: The input data. Is a torch tensor.
        
    Returns:
        Data with transformations applied. Since only normalisation has been applied
        the shape of the data will be the same."""
    
    logger = get_logger(f"Prepare data.")

    logger.info("Calculate mean and standard deviation.")
    mean_val, std_val = calc_mean_std(input_data)
    logger.info(
        f"Mean of colour channels: {mean_val}"
        f"Standard deviation of colour channels: {std_val}"
    )
    
    transformations = transforms.Compose([
        transforms.Normalize(mean_val, std_val)
    ])

    logger.info("Apply normalise tranformation")
    transformed_data = transformations(input_data)
    
    return transformed_data

class CustomDataset(Dataset):
    """Custom PyTorch dataset to feed in dataloader."""
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the data and label at specified index.

        img = self.data[idx]
        label = self.labels[idx]

        return img, label

def create_dataloader(input_data, labels, config):
    """Convert dataset to a PyTorch dataloader object.
    
    Args:
        input_data: The input data to convert.
        labels: The corresponding labels to the input data.
        config: Configuration file. The params.yaml file.
    
    Returns:
        A dataloader PyTorch object and data shape."""

    # Convert to custom dataset object to satisfy dataloader requirements.
    # Have __len__ and __getitem__ methods.

    logger = get_logger("Setup Dataloader")

    logger.info("Loading data into CustomDataset class.")
    dataset = CustomDataset(input_data, labels.to(long))

    data_shape = input_data.shape

    batchsize = config["prepare_data"]["batchsize"]
    shuffle = config["prepare_data"]["shuffle"]

    logger.info(f"Create data loader from custom dataset with batchsize: {batchsize}, shuffle: {shuffle}")
    dataloader = DataLoader(dataset, batch_size=config["prepare_data"]["batchsize"] , shuffle=config["prepare_data"]["shuffle"])

    return dataloader, data_shape