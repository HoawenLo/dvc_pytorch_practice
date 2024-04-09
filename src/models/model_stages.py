import datetime
import pickle

import torch
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix,  f1_score, precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve

from src.logging.log import get_logger

def train(optimiser, model, loss_fn, train_loader, device):
    """Train the model.
    
    Args:
        optimiser: The optimiser used.
        model: The trained model used to calculate the accuracy.
        loss_fn: The loss function.
        train_loader: The train dataset dataloader.
        val_loader: The validation dataset dataloader.
        device: Either CPU or GPU.
        
    Returns:
        Training and validation loss."""

    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(imgs)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Reset grad vals.
        optimiser.zero_grad()

        # Perform back propagation
        loss.backward()

        # Update weights
        optimiser.step()

        # Sum the losses over each iteration of the epoch and conver to python number
        # to avoid Pytorch autograd being applied to it.
        train_loss += loss.item()
    train_loss /= len(train_loader)

    return train_loss

def validate(model, loss_fn, val_loader, device):
    """Calculate validation loss.
    
    Args:
        model: The trained model used to calculate the accuracy.
        loss_fn: The loss function.
        val_loader: The validation dataloader.
        device: Either CPU or GPU.
        
    Returns:
        Returns validation loss value."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(imgs)

            # Calculate loss
            loss = loss_fn(outputs, labels)

            # Calculate validation loss
            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss


def accuracy(model, data_loader, device):
    """Calculate the accuracy of the model.
    
    Args:
        model: The trained model used to calculate the accuracy.
        data_loader: The dataset dataloader.
        device: Either CPU or GPU.
        
    Returns:
        Accuracy of the model with the particular dataloader."""

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy

def evaluate(model, data_loader, device):
    """Calculate the accuracy, precision, recall and f1 score of the model.
    
    Args:
        model: The trained model used to calculate the accuracy.
        data_loader: The dataset dataloader.
        device: Either CPU or GPU.
        
    Returns:
        Accuracy of the model with the particular dataloader."""

    correct = 0
    total = 0

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(predicted_labels, true_labels, average="macro")
    recall = recall_score(predicted_labels, true_labels, average="macro")
    f1 = f1_score(predicted_labels, true_labels, average="macro")

    return accuracy, precision, recall, f1

def return_metrics(predictions, true_labels, verbose=True):
    """Return accuracy, recall, precision and f1 scores.
    
    Args:
        predictions: The predicted classes.
        true_labels: The ground truth labels.
        verbose: Print out the metric values.
        
    Return:
        Accuracy, precision, recall, f1 score."""
    
    logger = get_logger("Metrics")

    logger.info("Calculating accuracy, precision, recall and f1 score.")
    accuracy = accuracy_score(predictions, true_labels)
    precision = precision_score(predictions, true_labels, average="macro")
    recall = recall_score(predictions, true_labels, average="macro")
    f1 = f1_score(predictions, true_labels, average="macro")

    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1

def export_pickle(input):
    
    with open("metric_vals.pkl", "wb") as pickle_file:
        pickle.dump(input, pickle_file)

def training_loop(n_epochs, optimiser, model, loss_fn, train_loader, val_loader, training_log_verboseness, device, data_shape):
    """Run a training loop for model for n number of epochs. 
    Load in the data with dataloader then perform a forward pass.
    Calculate the loss, perform back propagation then update the parameters.
    Apply the same with validation dataset.
    
    Args:
        n_epochs: Number of epochs, is an integer.
        optimiser: The chosen optimiser such as SGD or Adam.
        model: The input model to perform the training with.
        loss_fn: The loss function.
        train_loader: The dataloader for the training data.
        val_loader: The dataloader for the validation data.
        training_log_verboseness: The epoch multiple in which the loss and accuracy should be displayed.
        device: Either CPU or GPU.
        data_shape: The shape of the data. Used for torchsummary.summary.
        
    Returns:
        Train loss and accuracy, validation loss and accuracy over each epoch as lists."""

    train_loss_vals = []
    train_acc_vals = []
    val_loss_vals = []
    val_acc_vals = []
    val_recall_vals = []
    val_precision_vals = []
    val_f1_vals = []

    model = model.to(device)

    logger = get_logger("Train")

    logger.info("Providing model summary")
    summary(model, input_size=data_shape[1:])

    logger.info(
        f"Training model with optimiser: {optimiser}\n"
        f"with loss function: {loss_fn}\n"
        f"with number of epochs: {n_epochs}\n"
        f"with training_log_verboseness: {training_log_verboseness}\n"
        f"Training device: {device}"
    )
    logger.info("Commencing training.")
    for epoch in range(1, n_epochs + 1):
        train_loss = train(optimiser, model, loss_fn, train_loader, device)
        val_loss = validate(model ,loss_fn, val_loader, device)
        train_acc = accuracy(model, train_loader, device)
        val_acc, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)

        train_loss_vals.append(train_loss)
        train_acc_vals.append(train_acc)
        val_loss_vals.append(val_loss)
        val_acc_vals.append(val_acc)
        val_precision_vals.append(val_precision)
        val_recall_vals.append(val_recall)
        val_f1_vals.append(val_f1)

        if epoch == 1 or epoch % training_log_verboseness  == 0:
            print(f"{datetime.datetime.now()}, Epoch {epoch}, Training loss {train_loss}, Validation loss {val_loss}, Training Accuracy {train_acc}, Validation Accuracy {val_acc}")

    logger.info("Training complete.")

    export_pickle((train_loss_vals, train_acc_vals, val_loss_vals, val_acc_vals, val_recall_vals, val_precision_vals, val_f1_vals))

    return train_loss_vals, train_acc_vals, val_loss_vals, val_acc_vals


    