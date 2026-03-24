import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.train import FEATURES


def predict_churn(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza predicciones de abandono sobre un DataFrame de abonados.

    Parámetros:
        model: Modelo entrenado (sklearn).
        scaler: Scaler ajustado durante el entrenamiento (StandardScaler).
        df (pd.DataFrame): DataFrame con los datos de los abonados.
                           Debe contener las columnas de FEATURES, 'Abandono' e 'IdPersona'.

    Retorna:
        pd.DataFrame: DataFrame con columnas IdPersona, y_true, y_pred, y_prob, nivel_riesgo.
    """
    print("\n🧪 Inferencia datos nuevos...")

    X_val = df[FEATURES]
    y_val = df['Abandono']
    ids_persona = df['IdPersona']

    X_scaled = scaler.transform(X_val)

    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # --- Métricas ---
    acc = accuracy_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_val = roc_auc_score(y_val, y_prob)
    cm = confusion_matrix(y_val, y_pred)

    print(f"  Accuracy : {acc:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1       : {f1:.3f}")
    print(f"  AUC      : {auc_val:.3f}")
    print(f"  Matriz de confusión:\n{cm}")

    # --- Importancias globales ---
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        feature_importances = model.coef_.flatten()
    else:
        feature_importances = None

    if feature_importances is not None:
        df_importances_global = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": feature_importances
        }).sort_values("Importance", ascending=False)
        df_importances_global.to_csv("importancias_global.csv", index=False)

        # --- Importancias por persona ---
        # FIX: solo se calcula si feature_importances no es None
        importances_df = pd.DataFrame(X_scaled, columns=FEATURES)
        for i, f in enumerate(FEATURES):
            importances_df[f + '_importance'] = X_scaled[:, i] * feature_importances[i]
        importances_df['IdPersona'] = ids_persona.values
        importances_df.to_csv("importancias_persona.csv", index=False)
    else:
        print("⚠️ El modelo no tiene importancias de características.")

    # --- Predicciones ---
    pred_df = pd.DataFrame({
        "IdPersona": ids_persona.values,
        "y_true": y_val.values,
        "y_pred": y_pred,
        "y_prob": y_prob
    })
    pred_df['nivel_riesgo'] = pd.cut(
        pred_df['y_prob'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]
    )

    return pred_df
