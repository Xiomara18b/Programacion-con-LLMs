import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# --- FUNCIÓN DE LA MISIÓN ---
def analizar_estabilidad_coeficientes(X, y, n_bootstrap=100):
    """
    Realiza un análisis de estabilidad de coeficientes usando Bootstrap y Lasso.
    """
    coefs_acumulados = []
    n_filas = X.shape[0]
    
    # 1. Remuestreo (Numpy)
    for _ in range(n_bootstrap):
        # Generar índices aleatorios con reemplazo
        indices = np.random.choice(n_filas, size=n_filas, replace=True)
        X_res = X.iloc[indices]
        y_res = y.iloc[indices]
        
        # 2. Modelado (Sklearn) + Escalado
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)
        
        # Lasso con alpha=0.1
        modelo = Lasso(alpha=0.1)
        modelo.fit(X_scaled, y_res)
        
        # Almacenar coeficientes
        coefs_acumulados.append(modelo.coef_)
        
    # 3. Recolección de Coeficientes (Pandas)
    df_coefs = pd.DataFrame(coefs_acumulados, columns=X.columns)
    
    # 4. Cálculo de Robustez (Numpy/Pandas)
    # CV = Desviación Estándar / Media
    medias = df_coefs.mean()
    desviaciones = df_coefs.std()
    cv_valores = desviaciones / medias
    
    # 5. Retorno de DataFrame ordenado de menor a mayor CV
    resultado = pd.DataFrame({
        'variable': X.columns,
        'cv': cv_valores.values
    })
    
    return resultado.sort_values(by='cv').reset_index(drop=True)

# --- GENERADOR DE CASOS DE USO ---
def generar_caso_de_uso_analizar_estabilidad_coeficientes():
    """
    Genera un par input/output aleatorio para pruebas.
    """
    n_muestras = 100
    n_vars = 5
    
    # Creamos variables aleatorias
    X_data = np.random.randn(n_muestras, n_vars)
    columnas = [f'Factor_{i}' for i in range(n_vars)]
    X = pd.DataFrame(X_data, columns=columnas)
    
    # Creamos 'y' con una relación fuerte con Factor_0 (será la más estable)
    y = 5 * X['Factor_0'] + 2 * X['Factor_1'] + np.random.normal(0, 1, n_muestras)
    
    input_dict = {
        "X": X,
        "y": pd.Series(y),
        "n_bootstrap": 100
    }
    
    # Calculamos la salida esperada
    output_expected = analizar_estabilidad_coeficientes(**input_dict)
    
    return input_dict, output_expected

# --- BLOQUE DE SALIDA EN CONSOLA ---
if __name__ == "__main__":
    # Obtenemos los datos del generador
    input_test, output_test = generar_caso_de_uso_analizar_estabilidad_coeficientes()
    
    print("--- RESULTADO DE ESTABILIDAD (ORDENADO POR CV) ---")
    # Imprime el DataFrame final en la consola
    print(output_test)
    
    print("\nInterpretación: La variable al principio de la lista tiene el CV más bajo")
    print("y es la más confiable para el modelo de riesgo.")
