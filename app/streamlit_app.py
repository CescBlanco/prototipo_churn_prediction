import streamlit as st
import pandas as pd
import joblib
import os
import sys
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import predict_churn


# --- Carga de modelo y scaler ---
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("model/best_model.pkl")
    scaler = joblib.load("model/scaler.pkl")  # FIX: ruta consistente con train.py
    return model, scaler


model, scaler = load_model_and_scaler()

# --- Configuración de la página ---
st.set_page_config(page_title="App de Predicción de Abandono", layout="wide")

LOGO_GYM = "app/cem_horta-removebg-preview.png"
LOGO_AYUNTAMIENTO = "app/LOGO-AJUNTAMENT.png"

col1, colspace, col3 = st.columns([1, 3, 1])
with col1:
    st.image(LOGO_GYM, width=175)
with col3:
    st.image(LOGO_AYUNTAMIENTO, width=175)

st.markdown(
    "<h1 style='text-align: center; color: #66BB6A;'>Predicción de Abandono: CEM Horta Esportiva</h1>",
    unsafe_allow_html=True
)

# --- Tabs ---
tabs = st.tabs([":memo: Múltiples abonados", ":mag: Valoración predicción"])

# ── Tab 1: Predicción múltiple ──────────────────────────────────────────────
with tabs[0]:
    st.markdown("<h2 style='color: #888;'>Predicción de los abonados</h2>", unsafe_allow_html=True)

    st.info("""
        Sube los archivos necesarios con los datos de los abonados.
        Asegúrate de que contienen la información requerida (visitas, clases, etc.).
        El archivo principal debe llamarse **dataframe_final_abonado.csv**.
    """)

    uploaded_files = st.file_uploader(
        "Sube los archivos necesarios",
        type=["csv"],
        accept_multiple_files=True
    )

    if not uploaded_files:
        st.warning("Por favor, sube al menos un archivo para continuar.")
        st.stop()

    # Leer y mostrar archivos subidos
    files_data = {}
    for file in uploaded_files:
        df = pd.read_csv(file)
        files_data[file.name] = {
            "data": df,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filas": len(df),
            "columnas": len(df.columns)
        }

    st.write("### Archivos cargados:")
    # FIX: mostrar resumen como tabla, no el dict directamente
    resumen = pd.DataFrame([
        {"Archivo": name, "Filas": info["filas"], "Columnas": info["columnas"], "Hora subida": info["upload_time"]}
        for name, info in files_data.items()
    ])
    st.dataframe(resumen, use_container_width=True)

    # Validar que el archivo principal esté presente
    if "df_validacion.csv" not in files_data:
        st.error("❌ Falta el archivo principal: **df_validacion.csv**")
        st.stop()

    df_final = files_data["df_validacion.csv"]["data"]

    # Botón de predicción
    if st.button("🚀 Iniciar predicción abandono"):
        with st.spinner("Calculando predicciones..."):
            try:
                # FIX: se pasan model, scaler Y el dataframe
                df_predicciones = predict_churn(model, scaler, df_final)
                st.success("✅ Predicción completada")
                st.dataframe(df_predicciones, use_container_width=True)

                # Descarga de resultados
                csv = df_predicciones.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Descargar predicciones (.csv)",
                    data=csv,
                    file_name="predicciones_abandono.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Error durante la predicción: {e}")

# ── Tab 2: Valoración de la predicción ─────────────────────────────────────
with tabs[1]:
    st.markdown("<h2 style='color: #888;'>Valoración de la predicción</h2>", unsafe_allow_html=True)

    st.info("""
        En esta sección puedes revisar las métricas de rendimiento del modelo
        una vez se haya realizado la predicción.
    """)

    # Comprobar si ya se han generado predicciones en la sesión actual
    if "df_predicciones" not in st.session_state:
        st.warning("Primero realiza una predicción en la pestaña **Múltiples abonados**.")
    else:
        df_pred = st.session_state["df_predicciones"]

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total abonados", len(df_pred))
        col_b.metric("Predichos como abandono", int(df_pred["y_pred"].sum()))
        col_c.metric("% Abandono", f"{df_pred['y_pred'].mean() * 100:.1f}%")
        col_d.metric("Riesgo alto/muy alto",
                     int(df_pred["nivel_riesgo"].isin(["Alto", "Muy alto"]).sum()))

        st.write("### Distribución por nivel de riesgo")
        riesgo_counts = df_pred["nivel_riesgo"].value_counts().reset_index()
        riesgo_counts.columns = ["Nivel de riesgo", "Cantidad"]
        st.dataframe(riesgo_counts, use_container_width=True)
