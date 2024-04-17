import os
import pickle
import time
import yaml

from ..logging.log import get_logger

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
    logger.info(f"Export completed.")



    
