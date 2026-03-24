from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from typing import List, Dict


def calcular_metricas(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict[str, float]:
    """
    Calcula las métricas de rendimiento más comunes para un conjunto de predicciones.

    Parámetros:
        y_true (List[int]): Etiquetas verdaderas.
        y_pred (List[int]): Etiquetas predichas por el modelo.
        y_prob (List[float]): Probabilidades predichas para la clase positiva.

    Retorna:
        Dict[str, float]: Diccionario con accuracy, recall, f1_score y auc.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob)
    }
