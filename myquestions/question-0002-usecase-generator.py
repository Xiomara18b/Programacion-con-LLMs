import pandas as pd
import numpy as np
import random
from sklearn.ensemble import IsolationForest

# --- 1. GENERADOR DE CASOS DE USO ---
def generar_caso_de_uso_detectar_intrusiones_temporales():
    """
    Genera un caso de uso aleatorio (input/output) para la función de detección.
    """
    # Configuración aleatoria del tamaño del test
    n_samples = random.randint(50, 150)
    contam_val = random.uniform(0.01, 0.05)
    
    # Creación de datos sintéticos: Hora (0-23) y Presión (bar)
    horas = [random.randint(0, 23) for _ in range(n_samples)]
    
    # Simulamos una presión que depende un poco de la hora + ruido
    # (Más presión al mediodía, menos de madrugada)
    presiones = [
        abs(10 * np.sin(np.pi * h / 24) + random.uniform(0, 2)) 
        for h in horas
    ]
    
    df_input = pd.DataFrame({
        'hora': horas,
        'presion': presiones
    })
    
    # El Input es el diccionario de argumentos
    input_dict = {
        "df": df_input.copy(),
        "contamination": contam_val
    }
    
    # El Output esperado es el resultado de ejecutar la lógica sobre ese input
    # Se llama a la función definida más abajo
    output_expected = detectar_intrusiones_temporales(df_input.copy(), contamination=contam_val)
    
    return input_dict, output_expected

# --- 2. FUNCIÓN DE DETECCIÓN ---
def detectar_intrusiones_temporales(df, contamination=0.01):
    df_proc = df.copy()
    
    # 1. Ingeniería de Características Cíclicas
    # Aplicamos las fórmulas: sin(2 * pi * hora / 24) y cos(2 * pi * hora / 24)
    df_proc['hora_sin'] = np.sin(2 * np.pi * df_proc['hora'] / 24)
    df_proc['hora_cos'] = np.cos(2 * np.pi * df_proc['hora'] / 24)
    
    # 2. Preparación de Datos
    # Eliminamos la columna original para evitar redundancia lineal
    X = df_proc.drop(columns=['hora'])
    
    # 3. Detección de Anomalías con Isolation Forest
    model = IsolationForest(contamination=contamination, random_state=42)
    # Fit y Predict (Isolation Forest devuelve -1 para anomalías y 1 para normales)
    predicciones = model.fit_predict(X)
    
    # 4. Retorno: Nueva columna booleana (True si es -1)
    df['es_anomalia'] = predicciones == -1
    return df

# --- 3. EJECUCIÓN Y VERIFICACIÓN ---
# Generar un caso
params, resultado_esperado = generar_caso_de_uso_detectar_intrusiones_temporales()

# Probar la función del usuario
mi_resultado = detectar_intrusiones_temporales(**params)

# Verificación de salida
anomalias_detectadas = mi_resultado['es_anomalia'].sum()
print(f"Dataset generado con {len(mi_resultado)} registros.")
print(f"Anomalías detectadas: {anomalias_detectadas}")
