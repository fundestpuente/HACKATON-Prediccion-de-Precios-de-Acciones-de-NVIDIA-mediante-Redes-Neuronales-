import pandas as pd
import numpy as np

def agregar_indicadores_tecnicos(df):
    """
    Calcula RSI y MACD y los agrega al DataFrame.
    Asume que el df tiene una columna 'close'.
    """
    df = df.copy()
    
    # 1. RSI (Relative Strength Index) - Ventana estándar de 14 periodos
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Llenar NaNs del inicio (o podrías borrarlos después)
    df['rsi'] = df['rsi'].fillna(50) 

    # 2. MACD (Moving Average Convergence Divergence)
    # EMA rápida (12), EMA lenta (26)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # 3. (Opcional) Volumen - Si tu CSV lo tiene, úsalo. Si no, omite esta línea.
    # df['log_volume'] = np.log(df['volume'] + 1) # Log para normalizar picos grandes

    return df