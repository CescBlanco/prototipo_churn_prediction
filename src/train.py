import os
import time
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import joblib
import sys
import os
import logging

logging.basicConfig(level=logging.INFO)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.plotting import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from src.metrics import calcular_metricas

# Definición de las características
FEATURES = [
    'Edad', 'Sexo_Mujer', 'UsoServiciosExtra',
    'ratio_cantidad_2025_2024', 'Diversidad_servicios_extra',
    'TotalVisitas', 'TienePagos', 'DiasActivo',
    'VisitasUlt90', 'VisitasUlt180', 'TieneAccesos',
    'VisitasPrimerTrimestre', 'VisitasUltimoTrimestre',
    'DiaFav_domingo', 'DiaFav_jueves', 'DiaFav_lunes', 'DiaFav_martes',
    'DiaFav_miercoles', 'DiaFav_sabado', 'DiaFav_viernes',
    'EstFav_invierno', 'EstFav_otono', 'EstFav_primavera', 'EstFav_verano'
]


def separacion_df_inferencia(df: pd.DataFrame) -> tuple:
    """
    Separa el DataFrame en dos conjuntos: entrenamiento y validación.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos a dividir.

    Retorna:
        tuple: (df_validacion, df_entrenamiento)
    """
    df_0 = df[df['Abandono'] == 0]
    df_1 = df[df['Abandono'] == 1]

    n = int(0.10 * len(df))

    valid_0 = df_0.sample(n=n, random_state=42)
    valid_1 = df_1.sample(n=n, random_state=42)

    df_valid = pd.concat([valid_0, valid_1]).reset_index(drop=True)
    df_train = df.drop(df_valid.index).reset_index(drop=True)

    return df_valid, df_train


def cargar_datos(filepath: str, features: list) -> tuple:
    """
    Carga y prepara el conjunto de datos, separando en entrenamiento y validación.

    Parámetros:
        filepath (str): Ruta al archivo CSV con los datos.
        features (list): Lista de columnas a usar como características.

    Retorna:
        tuple: (X_train, y_train, scaler, df_valid)
    """
    df = pd.read_csv(filepath)
    df = df[df['Edad'] >= 18].reset_index(drop=True)
    df = df.rename(columns={
        'EsChurn': 'Abandono',
        'DiaFav_miércoles': 'DiaFav_miercoles',
        'DiaFav_sábado': 'DiaFav_sabado'
    })

    df_valid, df_train = separacion_df_inferencia(df)

    return df_train[features], df_train['Abandono'], df_valid


def main_train(filepath: str = 'dataframe_final_abonado.csv'):
    """
    Entrena múltiples modelos con GridSearch, selecciona el mejor por AUC
    y guarda el modelo y el scaler.

    Parámetros:
        filepath (str): Ruta al CSV de datos.

    Retorna:
        tuple: (resultados_df, best_model, scaler)
    """
    logging.info("Entrenando modelo...")

    X, y, df_valid = cargar_datos(filepath, FEATURES)

    # Guardar conjunto de validación
    df_valid.to_csv("df_validacion.csv", index=False)
    logging.info("Conjunto de validación guardado en df_validacion.csv")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Modelos
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier()
    }

    # Hiperparámetros
    param_grids = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10],
                          "min_samples_split": [2, 5], "min_samples_leaf": [1, 2]},
        "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1],
                              "max_depth": [2, 3, 4], "subsample": [0.8, 1.0]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf", "poly"], "gamma": ["scale", "auto"]},
        "KNN": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
    }

    resultados = []
    best_model, best_auc = None, 0.0

    for nombre, modelo in modelos.items():
        print(f"\n🔍 Entrenando modelo: {nombre}")
        start = time.time()

        grid = GridSearchCV(modelo, param_grids[nombre], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model_temp = grid.best_estimator_

        y_pred = best_model_temp.predict(X_test_scaled)
        y_prob = best_model_temp.predict_proba(X_test_scaled)[:, 1]

        metrics = calcular_metricas(y_test, y_pred, y_prob)

        # Gráficas
        plot_confusion_matrix(y_test, y_pred, f"cm_{nombre}.png")
        plot_roc_curve(y_test, y_prob, nombre, f"roc_curve_{nombre}.png")

        if hasattr(best_model_temp, 'feature_importances_') or hasattr(best_model_temp, 'coef_'):
            plot_feature_importance(best_model_temp, list(X_train.columns), f"feat_imp_{nombre}.png")
        else:
            print(f"⚠️ El modelo {nombre} no tiene importancias de características.")

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_model = best_model_temp

        resultados.append({
            "Modelo": nombre,
            **metrics,
            "Tiempo": round(time.time() - start, 2)
        })

    # FIX: resultados finales FUERA del bucle
    resultados_df = pd.DataFrame(resultados).sort_values(by="auc", ascending=False)
    print("\n🏁 Resultados finales:\n", resultados_df)
    print(f"\n✅ Mejor modelo: {best_model.__class__.__name__} con AUC: {best_auc:.3f}")

    return resultados_df, best_model, scaler


if __name__ == "__main__":
    resultados_df, best_model, scaler = main_train()

    # FIX: guardado DENTRO del bloque __main__, con rutas consistentes
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/best_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print("✅ Modelo y scaler guardados en model/")
