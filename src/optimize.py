import numpy as np
from scipy.optimize import linprog

def optimizar_posicion(precio_actual, precio_predicho, presupuesto, riesgo_maximo, volatilidad_actual):
    """
    Resuelve un problema de Programación Lineal Entera (o relajada) para decidir cuántas acciones comprar.
    
    Maximizar Z = (Precio_Predicho - Precio_Actual) * x
    Sujeto a:
        1. (Precio_Actual * x) <= Presupuesto
        2. (Volatilidad * x) <= Riesgo_Maximo (medida simplificada de exposición al riesgo)
        3. x >= 0
    """
    
    retorno_esperado = precio_predicho - precio_actual
    
    # Si la predicción es negativa o neutra, la recomendación es 0 (o Vender si tuviéramos stock)
    if retorno_esperado <= 0:
        return {
            'accion': 'MANTENER / VENDER',
            'cantidad': 0,
            'razon': 'Tendencia bajista o neutra detectada.'
        }

    # === Configuración del Solver (Simplex/Interior-Point) ===
    # linprog minimiza, así que usamos el negativo del retorno para maximizar
    c = [-retorno_esperado] 
    
    # Restricciones de Desigualdad (A_ub * x <= b_ub)
    # 1. Restricción de Presupuesto: Precio * x <= Presupuesto
    # 2. Restricción de Riesgo: Volatilidad * x <= Riesgo_Maximo
    A = [
        [precio_actual], 
        [volatilidad_actual]
    ]
    b = [presupuesto, riesgo_maximo]
    
    # Límites de las variables (x >= 0)
    x_bounds = (0, None)
    
    # Resolver
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds], method='highs')
    
    if res.success:
        cantidad_optima = int(res.x[0]) # Convertimos a entero (acciones completas)
        return {
            'accion': 'COMPRAR',
            'cantidad': cantidad_optima,
            'inversion_estimada': cantidad_optima * precio_actual,
            'retorno_proyectado': cantidad_optima * retorno_esperado,
            'razon': 'Oportunidad de ganancia detectada dentro de los parámetros de riesgo.'
        }
    else:
        return {
            'accion': 'ERROR',
            'cantidad': 0,
            'razon': 'No se encontró solución óptima.'
        }