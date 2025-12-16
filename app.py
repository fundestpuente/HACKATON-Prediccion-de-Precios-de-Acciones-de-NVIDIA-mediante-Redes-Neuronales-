import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.optimize import optimizar_posicion
from src.indicators import agregar_indicadores_tecnicos # Tu función del Día 1

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="NVIDIA AI-Sight", layout="wide", page_icon="📈")

# Estilos CSS personalizados para el Hackathon
st.markdown("""
    <style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stButton>button {width: 100%; background-color: #76b900; color: white;}
    </style>
    """, unsafe_allow_html=True)

# === TÍTULO Y SIDEBAR ===
st.title("📈 Optimización de Inversiones con IA: Predicción de Precios de Acciones de NVIDIA")
st.markdown("Sistema de **Deep Learning (LSTM)** y **Optimización Lineal** para la toma de decisiones en NVDA.")

st.sidebar.header("⚙️ Panel de Control")
presupuesto = st.sidebar.number_input("Presupuesto Disponible (USD)", value=5000, step=100)
riesgo = st.sidebar.slider("Tolerancia al Riesgo (Factor Volatilidad)", 1.0, 100.0, 50.0)

# === CARGA DE DATOS Y MODELO ===
@st.cache_data
def load_data():
    # Asegúrate de que el CSV esté en la carpeta correcta
    df = pd.read_csv("valid_csv_concatenado_invertido_con_60min.csv", parse_dates=['timestamp'])
    df = df.sort_values('timestamp')
    df = agregar_indicadores_tecnicos(df) # Feature Engineering del Día 1
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_ai_model():
    # Aquí cargarías tu modelo entrenado. 
    # NOTA: Para el demo, si no tienes el .h5 listo, maneja la excepción.
    try:
        return load_model('models/mi_modelo_lstm.h5')
    except:
        return None

df = load_data()
model = load_ai_model()

# === VISUALIZACIÓN PRINCIPAL ===
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Análisis de Mercado en Tiempo Real")
    
    # Gráfico de Velas con Plotly
    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='NVDA')])
    
    # Agregar líneas de indicadores (opcional)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'].rolling(window=20).mean(), 
                             line=dict(color='orange', width=1), name='Media Móvil 20'))
    
    fig.update_layout(height=500, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# === PREDICCIÓN Y OPTIMIZACIÓN ===
with col2:
    st.subheader("🤖 AI Advisor")
    
    # Simulación de obtener el último dato real
    last_row = df.iloc[-1]
    precio_actual = last_row['close']
    rsi_actual = last_row['rsi']
    
    # --- PREDICCIÓN (Mockup o Real) ---
    # Si el modelo cargó, predecimos. Si no, usamos un dummy para probar la UI.
    if model:
        # Prepara los últimos 60 datos para predecir
        # input_data = ... (Lógica de scaling del Día 1)
        # precio_predicho = model.predict(input_data)
        precio_predicho = precio_actual * 1.02 # Placeholder si el modelo no carga hoy
    else:
        st.warning("Modelo .h5 no encontrado. Usando simulación.")
        precio_predicho = precio_actual * 1.025 # Simulación: Predice subida del 2.5%

    delta = precio_predicho - precio_actual
    color_delta = "normal" if delta > 0 else "off"
    
    st.metric(label="Precio Actual", value=f"${precio_actual:.2f}")
    st.metric(label="Predicción IA (T+1)", value=f"${precio_predicho:.2f}", delta=f"{delta:.2f} USD")
    
    st.markdown("---")
    
    # --- MÓDULO DE OPTIMIZACIÓN ---
    st.write("### Estrategia Sugerida")
    
    # Ejecutamos la función de optimización lineal
    volatilidad_proxy = (last_row['high'] - last_row['low']) # Simplificación
    resultado = optimizar_posicion(precio_actual, precio_predicho, presupuesto, riesgo, volatilidad_proxy)
    
    if resultado['accion'] == 'COMPRAR':
        st.success(f"🚀 **RECOMENDACIÓN: {resultado['accion']}**")
        st.write(f"Cantidad: **{resultado['cantidad']} acciones**")
        st.write(f"Inversión: ${resultado['inversion_estimada']:.2f}")
        st.info(f"💡 {resultado['razon']}")
    else:
        st.error(f"🛑 **RECOMENDACIÓN: {resultado['accion']}**")
        st.write(f"Motivo: {resultado['razon']}")

# === INFO DEL EQUIPO ===
st.markdown("---")
st.caption("Desarrollado por: Ulises, Esteban, Brayan, Alan y Sofia | Samsung Innovation Campus 2025")