import os
import pickle
import time
import json

import numpy as np

from ..logging.log import get_logger

def extract_metric_data(metric_data):
    """Extract the metrics from the metrics data dictionary and find the average
    for each metric.

    Args:
        metric_data: The dictionary which holds data on training results.

    Returns:
        None
    """

    train_loss_vals = np.array(metric_data["train_loss_vals"])
    train_acc_vals = np.array(metric_data["train_acc_vals"])
    val_loss_vals = np.array(metric_data["val_loss_vals"])
    val_acc_vals = np.array(metric_data["val_acc_vals"])

    average_train_loss = np.mean(train_loss_vals)
    average_train_acc = np.mean(train_acc_vals)
    average_val_loss = np.mean(val_loss_vals)
    average_val_acc = np.mean(val_acc_vals)

    return average_train_loss, average_train_acc, average_val_loss, average_val_acc

def create_json(metric_data, json_filepath):
    """Create a JSON file to store the results.

    Args: 
        metric_data: The dictionary which holds data on training results.
        json_filepath: Filepath to the results.json file.

    Returns:
        None
    """
    
    average_train_loss, average_train_acc, average_val_loss, average_val_acc = extract_metric_data(metric_data)

    data = {
        "average_train_loss": average_train_loss,
        "average_train_acc": average_train_acc,
        "average_val_loss": average_val_loss,
        "average_val_acc": average_val_acc
    }

    with open(json_filepath, "w") as file:
        json.dump(data, file, indent=4)

def export_metrics(config, metric_data):
    """Export the metric data to the metrics directory
    as a pickle file.
    
    Args:
        config: The configuration params.yaml file.
        
    Returns:
        None"""
    
    logger = get_logger("Export metrics")
    logger.info("Exporting training loss, validation loss, training accuracy and validation accuracy.")
    
    # Setup destination filepath.
    dst_filepath = config["export_data"]["dst_filepath"]
    logger.info(f"Destination filepath: {dst_filepath}")

    # Setup filename.
    logger.info(f"Creating filename.")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    filename = "metric_data_" + timestamp + ".pkl"
    temp_filename = "metric_data.pkl"
    logger.info(f"Filename: {filename}")

    # Combine filename and destination filepath.
    updated_dst_filepath = os.path.join(dst_filepath, filename)
    updated_dst_filepath_temp = os.path.join(dst_filepath, temp_filename)

    # Export pickle file to destination filepath.
    logger.info(f"Exporting {filename} to destination filepath {dst_filepath}.")
    with open(updated_dst_filepath, "wb") as output_file:
        pickle.dump(metric_data, output_file)
    with open(updated_dst_filepath_temp, "wb") as output_file:
        pickle.dump(metric_data, output_file)
    
    # Export results to yaml file.
    logger.info(f"Exporting to results.json to track metrics.")
    json_filepath = config["export_data"]["json_filepath"]
    create_json(metric_data, json_filepath)
    
    logger.info(f"Export completed.")



    
