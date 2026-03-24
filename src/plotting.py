import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from typing import List


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], path: str) -> None:
    """
    Dibuja y guarda la matriz de confusión como imagen.

    Parámetros:
        y_true (List[int]): Etiquetas verdaderas.
        y_pred (List[int]): Etiquetas predichas.
        path (str): Ruta donde se guardará la imagen.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_roc_curve(y_test, y_prob, model_name: str, output_path: str) -> None:
    """
    Genera y guarda la curva ROC para el modelo entrenado.

    Parámetros:
        y_test: Etiquetas reales del conjunto de test.
        y_prob: Probabilidades predichas para la clase positiva.
        model_name (str): Nombre del modelo (usado en el título).
        output_path (str): Ruta donde se guardará la imagen.
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(model, features: list, output_path: str) -> None:
    """
    Genera y guarda el gráfico de importancia de las variables.

    Parámetros:
        model: Modelo entrenado con `feature_importances_` o `coef_`.
        features (list): Nombres de las características.
        output_path (str): Ruta donde se guardará la imagen.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print(f"⚠️ El modelo {model.__class__.__name__} no tiene importancias de características.")
        return

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Importancia de las Características - {model.__class__.__name__}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
