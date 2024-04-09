import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix,  f1_score, precision_score, precision_recall_curve, recall_score, roc_auc_score, roc_curve,

from src.logging.log import get_logger

def confusion_matrix(predictions, true_labels, normalise=False):
    """Create a confusion matrix.
    
    Args:
        predictions: Class predictions.
        true_labels: The ground truth labels.
        normalise: Whether to use normalised values or not for the confusion matrix.
        
    Return:
        A sci-kit learn confusion matrix."""
    
    scale = [0, 1]
    tick_labels = ["No", "Yes"]

    if normalise:
        conf_mat = confusion_matrix(predictions, true_labels, labels=scale, normalize="true")
    else:
        conf_mat = confusion_matrix(predictions, true_labels, labels=scale)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap="coolwarm")
    fig.colorbar(cax)
    ax.set_xticklabels([''] + tick_labels)
    ax.set_yticklabels([''] + tick_labels)
    plt.xlabel('Predicted')
    ax.xaxis.set_label_position('top') 
    plt.ylabel('True')
    plt.show()

    return conf_mat

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

def multiclass_precision_recall_roc(true_labels, predictions):
    """Compute precision-recall curves and ROC curves for multiclass classification.
    Args:
        true_labels (tensor-like): True binary labels in binary indicator matrix format.
        Shape: (n_samples, n_classes).
        predictions (tensor-like): Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions (as returned
        by "decision_function" on some classifiers).
        Shape: (n_samples, n_classes).

    Returns:
        pr_auc (dict): Dictionary containing area under the precision-recall curve (PR AUC) for each class.
        pr_auc_micro (float): Micro-average area under the precision-recall curve (PR AUC).
        roc_auc (dict): Dictionary containing area under the ROC curve (ROC AUC) for each class.
        roc_auc_micro (float): Micro-average area under the ROC curve (ROC AUC)."""
    true_labels = true_labels.cpu().detach().numpy() if true_labels.is_cuda else true_labels.numpy()
    predictions = predictions.cpu().detach().numpy() if predictions.is_cuda else predictions.numpy()

    # Precision-Recall curve
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(true_labels.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], predictions[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Compute micro-average precision-recall curve and area under the curve
    precision_micro, recall_micro, _ = precision_recall_curve(true_labels.ravel(), predictions.ravel())
    pr_auc_micro = auc(recall_micro, precision_micro)

    # ROC curve and ROC AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(true_labels.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(true_labels.ravel(), predictions.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    return pr_auc, pr_auc_micro, roc_auc, roc_auc_micro

def plot_multiclass_precision_recall_roc(pr_auc, pr_auc_micro, roc_auc, roc_auc_micro, recall, precision, fpr, tpr, recall_micro, precision_micro, fpr_micro, tpr_micro):
    
    """Plot precision-recall curves and ROC curves for multiclass classification.

    Args:
        pr_auc (dict): Dictionary containing area under the precision-recall curve (PR AUC) for each class.
        pr_auc_micro (float): Micro-average area under the precision-recall curve (PR AUC).
        roc_auc (dict): Dictionary containing area under the ROC curve (ROC AUC) for each class.
        roc_auc_micro (float): Micro-average area under the ROC curve (ROC AUC).
        recall (dict): Recall values for each class.
        precision (dict): Precision values for each class.
        fpr (dict): False positive rate values for each class.
        tpr (dict): True positive rate values for each class.
        recall_micro (array-like): Micro-average recall values.
        precision_micro (array-like): Micro-average precision values.
        fpr_micro (array-like): Micro-average false positive rate values.
        tpr_micro (array-like): Micro-average true positive rate values.
        """
    num_classes = len(pr_auc)

    # Plot Precision-Recall curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'Class {i} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Plot ROC curves
    plt.subplot(1, 2, 2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Plot micro-average Precision-Recall and ROC curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(recall_micro, precision_micro, label=f'Micro-average (AUC = {pr_auc_micro:.2f})', color='black')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Micro-average Precision-Recall Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average (AUC = {roc_auc_micro:.2f})', color='black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()




