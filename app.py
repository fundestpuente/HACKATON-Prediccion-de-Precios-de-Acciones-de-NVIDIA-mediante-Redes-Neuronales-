import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from src.optimize import optimizar_posicion
from src.indicators import agregar_indicadores_tecnicos

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="NVIDIA AI-Sight", layout="wide")

# Estilos CSS personalizados
st.markdown("""
    <style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;}
    .stButton>button {width: 100%; background-color: #76b900; color: white;}
    </style>
    """, unsafe_allow_html=True)

# === TÍTULO ===
st.title("NVIDIA AI-Sight: Inversión Inteligente")
st.markdown("Sistema de **Deep Learning (LSTM)** y **Optimización Lineal** para la toma de decisiones en NVDA.")

# === SIDEBAR: CONTROLES ===
st.sidebar.header("Panel de Control")

# 1. Selección de Fuente de Datos
st.sidebar.subheader("Fuente de Datos")
modo_datos = st.sidebar.radio("Selecciona la fuente:", ("Simulación (CSV 2023)", "Mercado en Vivo (Yahoo Finance)"))

# 2. Parámetros de Inversión
st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros de Cartera")
presupuesto = st.sidebar.number_input("Presupuesto Disponible (USD)", value=5000, step=100)
riesgo = st.sidebar.slider("Tolerancia al Riesgo (Factor Volatilidad)", 1.0, 100.0, 50.0)

# === FUNCIÓN DE CARGA DE DATOS ===
@st.cache_data(ttl=600) # Cache de 10 minutos para no saturar Yahoo
def get_data(mode):
    if mode == "Mercado en Vivo (Yahoo Finance)":
        # Descarga datos recientes
        stock = yf.Ticker("NVDA")
        df = stock.history(period="1y", interval="1d")
        
        # Limpieza
        df.reset_index(inplace=True)
        df.rename(columns={'Date':'timestamp', 'Open':'open', 'High':'high', 'Low':'low', 'Close':'close', 'Volume':'volume'}, inplace=True)
        
        # Manejo de Zona Horaria
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        
        # Feature Engineering (Calcula RSI, MACD al vuelo)
        df = agregar_indicadores_tecnicos(df)
        df.dropna(inplace=True)
        return df
    else:
        # Carga CSV local (Simulación)
        df = pd.read_csv("valid_csv_concatenado_invertido_con_60min.csv", parse_dates=['timestamp'])
        df = df.sort_values('timestamp')
        df = agregar_indicadores_tecnicos(df)
        df.dropna(inplace=True)
        return df

# === CARGA DEL MODELO IA ===
@st.cache_resource
def load_ai_model():
    try:
        return load_model('models/mi_modelo_lstm.h5')
    except:
        return None

# Ejecutar cargas
with st.spinner('Conectando con los mercados...'):
    df = get_data(modo_datos)
    model = load_ai_model()

# === VISUALIZACIÓN PRINCIPAL ===
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader(f"📊 Análisis Técnico: {modo_datos}")
    
    # Gráfico de Velas Interactivo
    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'], name='NVDA')])
    
    # Personalización del gráfico
    fig.update_layout(height=500, template="plotly_dark", title_text="Histórico de Precios NVDA")
    fig.update_layout(xaxis_rangeslider_visible=False) # Ocultar slider inferior para limpieza
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Advisor")
    
    if df.empty:
        st.error("No hay datos disponibles.")
        st.stop()

    # Obtener el ÚLTIMO dato disponible
    last_row = df.iloc[-1]
    precio_actual = last_row['close']
    
    # --- PREDICCIÓN CON EL MODELO ---
    precio_predicho = precio_actual # Valor por defecto

    if len(df) >= 60 and model is not None:
        try:
            # 1. Extraer ventana de 60 días (Close, RSI, MACD)
            ultimos_60_dias = df.iloc[-60:][['close', 'rsi', 'macd']].values
            
            # 2. Escalar (Ajustamos scaler a esta ventana para la demo)
            scaler_live = MinMaxScaler()
            input_scaled = scaler_live.fit_transform(ultimos_60_dias)
            
            # 3. Reshape para LSTM (1 muestra, 60 pasos, 3 features)
            input_reshaped = np.reshape(input_scaled, (1, 60, 3))
            
            # 4. Predecir
            prediccion_scaled = model.predict(input_reshaped, verbose=0)
            
            # 5. Invertir escala (Scaler auxiliar solo para precio)
            scaler_precio = MinMaxScaler()
            scaler_precio.fit(df[['close']].iloc[-60:])
            precio_predicho = scaler_precio.inverse_transform(prediccion_scaled)[0][0]
            
        except Exception as e:
            st.warning(f"Error en predicción: {e}")
    else:
        if model is None:
            st.warning("⚠️ Modelo .h5 no encontrado.")
        else:
            st.warning("⚠️ Insuficientes datos para ventana de 60 días.")

    # --- MÉTRICAS DE RESULTADO ---
    delta = precio_predicho - precio_actual
    
    st.metric(label="Precio Cierre (Último)", value=f"${precio_actual:.2f}")
    
    # Lógica de color para la predicción
    st.metric(
        label="Predicción IA (Tendencia)", 
        value=f"${precio_predicho:.2f}", 
        delta=f"{delta:.2f} USD", 
        delta_color="normal"
    )
    
    st.markdown("---")
    
    # --- MÓDULO DE OPTIMIZACIÓN (Tu diferenciador) ---
    st.write("### Estrategia Sugerida")
    
    # Proxy de volatilidad (Rango del día)
    volatilidad_proxy = (last_row['high'] - last_row['low']) 
    
    resultado = optimizar_posicion(precio_actual, precio_predicho, presupuesto, riesgo, volatilidad_proxy)
    
    if resultado['accion'] == 'COMPRAR':
        st.success(f"🚀 **RECOMENDACIÓN: {resultado['accion']}**")
        st.write(f"🎯 Cantidad: **{resultado['cantidad']} acciones**")
        st.write(f"💵 Inversión: **${resultado['inversion_estimada']:.2f}**")
        st.caption(f"💡 Razón: {resultado['razon']}")
    elif resultado['accion'] == 'MANTENER / VENDER':
        st.warning(f"✋ **RECOMENDACIÓN: {resultado['accion']}**")
        st.caption(f"💡 Razón: {resultado['razon']}")
    else:
        st.error("Error en cálculo de optimización.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado para Samsung Innovation Campus 2025 | Equipo NVIDIA AI-Sight")