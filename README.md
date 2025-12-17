# 📈 Optimización de Inversiones con IA: Predicción de precios de acciones de NVIDIA

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Prototipo_Hackathon-green)

> **Proyecto Final Samsung Innovation Campus – Módulo de Inteligencia Artificial (EC04)**

## 🚩 Problemática
[cite_start]NVIDIA (NVDA) es un activo de alto crecimiento, pero su volatilidad anual (~40%) dificulta la toma de decisiones para inversores minoristas[cite: 86, 88]. Las herramientas tradicionales (como medias móviles) reaccionan tarde a los cambios rápidos del mercado, generando incertidumbre y riesgo financiero.

## 🎯 Nuestra Solución
**NVIDIA AI-Sight** no solo predice el precio; democratiza el acceso a análisis institucional. Desarrollamos un sistema inteligente que combina:
1.  **Deep Learning (LSTM):** Para capturar patrones temporales complejos en el precio.
2.  **Análisis Multivariado:** Incorporamos indicadores técnicos (RSI, MACD) y volumen para robustecer la predicción.
3.  **Optimización de Cartera:** Un módulo prescriptivo que sugiere decisiones basadas en la predicción.

**Objetivo de Rendimiento:** MAPE (Error Porcentual Absoluto Medio) < 2%.

## 🚀 Características Clave (Roadmap Hackathon)

### 1. Modelo LSTM Multivariado
A diferencia de los modelos básicos, nuestro motor de IA se alimenta de:
* Precios históricos (OHLC).
* **Indicadores Técnicos:** RSI (Relative Strength Index) y MACD para detectar sobrecompra/sobreventa.
* **Volumen:** Para confirmar la fuerza de las tendencias.

### 2. Dashboard Interactivo (Streamlit)
Una interfaz web amigable para visualizar:
* Gráficos dinámicos de velas japonesas.
* Línea de predicción de la IA vs. Datos reales.
* Métricas de error en tiempo real.

### 3. Módulo de Optimización
Utilizando la salida del modelo LSTM, aplicamos algoritmos de optimización para responder a la pregunta: *"Dada esta predicción, ¿cuál es la exposición al riesgo sugerida?"*.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.10
* **Modelado:** TensorFlow / Keras (LSTM Layers, Dropout para evitar overfitting).
* **Procesamiento de Datos:** Pandas, NumPy, Scikit-learn (MinMaxScaler).
* **Visualización:** Matplotlib, Plotly (para gráficos interactivos).
* **Despliegue/Interfaz:** Streamlit.

## 👥 Equipo de Desarrollo

| Nombre | Rol |
| :--- | :--- |
| **Ulises Chingo** | Líder de Proyecto |
| **Esteban Quiña** | Analista de Procesamiento de Datos |
| **Brayan Maisincho** | Analista de Datos |
| **Alan Palma** | Analista del Modelo AI |
| **Sofia Feijóo** | Analista de Resultados |


## 📊 Estructura del Proyecto

```bash
├── data/               # Datasets (CSV original y procesado)
├── models/             # Archivos .h5 del modelo entrenado
├── src/                # Código fuente
│   ├── model_LSTM.py   # Arquitectura de la red neuronal
│   ├── indicators.py   # Cálculo de RSI, MACD, etc.
│   └── optimize.py     # Lógica de optimización de portafolio
├── app.py              # Aplicación principal (Streamlit)
├── requirements.txt    # Dependencias
└── README.md           # Documentación del proyecto
