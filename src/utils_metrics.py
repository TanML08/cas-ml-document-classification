import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def cls_metrics(y_true, y_pred, labels=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, digits=4, labels=labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)
