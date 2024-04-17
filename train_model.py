
import argparse

import yaml
from torch import nn, optim, cuda

from src.data.load_data import load_data
from src.data.prepare_data import apply_transformation, create_dataloader
from src.models.convnet import Model
from src.models.model_stages import training_loop
from src.logging.log import get_logger
from src.export_data.export import export_metrics

def train_model(config_path):
    """Master function. Loads in data, prepares data and trains model.
    
    Args:
        config_path: The filepath of params.yaml file.
        
    Returns:
        None"""

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Load in data
    train_imgs, train_labels, val_imgs, val_labels = load_data(config)

    # Prepare data
    train_imgs = apply_transformation(train_imgs)
    val_imgs = apply_transformation(val_imgs)

    train_dataloader, data_shape = create_dataloader(train_imgs, train_labels, config)
    val_dataloader, _ = create_dataloader(val_imgs, val_labels, config)

    # Train model
    logger = get_logger("Train")

    model_param = config["train"]["torch_params"]["model"]
    loss_param = config["train"]["torch_params"]["loss_fn"]
    optimiser_param = config["train"]["torch_params"]["optimiser"]

    logger.info(f"Initialise model: {model_param}")
    logger.info(f"Initialise loss function: {loss_param}")
    logger.info(f"Initialise optimiser: {optimiser_param}")
    model = Model()
    optimiser = optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    if cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    train_loss_vals, train_acc_vals, val_loss_vals, val_acc_vals = training_loop(
        n_epochs = config["train"]["n_epochs"],
        optimiser=optimiser,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        training_log_verboseness=config["train"]["training_log_verboseness"],
        device=device,
        data_shape=data_shape
    )

    metric_data = {"train_loss_val":train_loss_vals,
                   "train_acc_val":train_acc_vals,
                   "val_loss_vals":val_loss_vals,
                   "val_acc_vals":val_acc_vals}

    export_metrics(config, metric_data)

if __name__ == "__main__":
    # Create an parser
    parser = argparse.ArgumentParser("Train model")

    # Add config argument
    parser.add_argument("--config", dest="config", help="Params.yaml filepath", required=True)

    # Parse arguments
    args = parser.parse_args()
    
    train_model(config_path=args.config)
    



