import numpy as np
import joblib
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_recall_curve
import matplotlib.pyplot as plt


def plot(fpr, tpr, roc_auc, recall, precision, pr_auc, thresholds_roc, results_array=None):
    """
    Plot ROC and Precision-Recall curves and optionally print accuracy/recall
    for different thresholds.

    Parameters
    ----------
    fpr, tpr : array-like
    roc_auc : float
    recall, precision : array-like
    pr_auc : float
    thresholds_roc : array-like
    results_array : numpy.ndarray, optional
        If provided, calculates accuracy and recall at each ROC threshold.
    """

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()
    print("auc: ", roc_auc)
    print("pr auc: ", pr_auc)
    
    if results_array is not None:
        accuracies = []
        recalls = []
        for threshold in thresholds_roc:
            y_pred = (results_array[:, 0] >= threshold).astype(int)
            acc = accuracy_score(results_array[:, 1], y_pred)
            rec = recall_score(results_array[:, 1], y_pred)
            accuracies.append(acc)
            recalls.append(rec)
        print("acc: ", accuracies)
        print("recalls: ", recalls)
    
    
def read_model_preprocess_data(model_path, pca_path, df_numeric):
    """
    Load model and preprocessing object (e.g., PCA) from disk and transform data.

    Parameters
    ----------
    model_path : str
    pca_path : str
    df_numeric : pandas.DataFrame

    Returns
    -------
    estimator, transformed_data
    """
    
    estimator = joblib.load(model_path)
    pca = joblib.load(pca_path)
    if hasattr(pca, "transform"):
        df_numeric_pca = pca.transform(df_numeric)
    else:
        df_numeric_pca = df_numeric
    return estimator, df_numeric_pca

def validate_per_file(results_array):
    """
    Compute ROC and PR metrics for per-file validation results.
    """

    fpr, tpr, thresholds_roc = roc_curve(results_array[:, 1], results_array[:, 0])
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds_pr = precision_recall_curve(results_array[:, 1], results_array[:, 0])
    pr_auc = auc(recall, precision)
    return fpr, tpr, thresholds_roc, roc_auc, precision, recall, pr_auc

def validate_per_action(scores, is_illegal):
    """
    Compute ROC and PR metrics for per-action validation results.
    """
    fpr, tpr, thresholds_roc = roc_curve(is_illegal, scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds_pr = precision_recall_curve(is_illegal, scores)
    pr_auc = auc(recall, precision)
    return fpr, tpr, thresholds_roc, roc_auc, precision, recall, pr_auc

def collect_results_per_file(scores, file_id, is_illegal):
    results = []
    unique_files = np.unique(file_id)
    for file in unique_files:
        indices = (file_id == file)
        avg_distance = np.mean(scores[indices])
        is_legal_per_file = np.mean(is_illegal[indices])
        results.append([avg_distance, is_legal_per_file])
    results_array = np.array(results)
    return results_array

def visualize_results(scores, file_id, is_illegal):
    """
    Aggregate results per file, compute metrics, and plot ROC/PR curves.
    """    
    results_array = collect_results_per_file(scores, file_id, is_illegal)
    fpr, tpr, thresholds_roc, roc_auc, precision, recall, pr_auc = validate_per_file(results_array)
    plot(fpr, tpr, roc_auc, recall, precision, pr_auc, thresholds_roc, results_array)            