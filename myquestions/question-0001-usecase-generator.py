import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix

# --- 1. GENERADOR DE CASOS DE USO ---
def generar_caso_de_uso_auditar_equidad_genero():
    """
    Genera un caso aleatorio (input) y calcula el resultado esperado (output).
    """
    # Configuración aleatoria de n entre 15 y 50
    n = random.randint(15, 50)
    data = {
        'genero': [random.choice(['M', 'F']) for _ in range(n)],
        'real': [random.choice([0, 1]) for _ in range(n)],
        'prediccion': [random.choice([0, 1]) for _ in range(n)]
    }
    df_test = pd.DataFrame(data)
    
    # El 'output' se genera llamando a la función lógica definida abajo
    expected_output = auditar_equidad_genero(df_test)
    
    return {"df": df_test}, expected_output

# --- 2. FUNCIÓN DE AUDITORÍA ---
def auditar_equidad_genero(df):
    """
    Función objetivo: Calcula FNR por género y detecta disparidad.
    """
    # 1. Cálculo de métricas por grupo
    df_h = df[df['genero'] == 'M']
    df_m = df[df['genero'] == 'F']
    
    def obtener_fnr(sub_df):
        if sub_df.empty:
            return 0.0
        # 2. Uso de Sklearn (TN, FP, FN, TP)
        # Forzamos labels [0, 1] para que la matriz siempre sea 2x2
        tn, fp, fn, tp = confusion_matrix(sub_df['real'], sub_df['prediccion'], labels=[0, 1]).ravel()
        
        # 3. Cálculo de FNR: FN / (FN + TP)
        denominador = fn + tp
        return float(fn / denominador) if denominador > 0 else 0.0

    fnr_h = obtener_fnr(df_h)
    fnr_m = obtener_fnr(df_m)
    
    # 4. Detección de Disparidad (FNR_max / FNR_min)
    max_fnr = max(fnr_h, fnr_m)
    min_fnr = min(fnr_h, fnr_m)
    
    if min_fnr == 0:
        disparidad = float('inf') if max_fnr > 0 else 0.0
    else:
        disparidad = float(max_fnr / min_fnr)
    
    # 5. Retorno
    return {
        "fnr_hombres": fnr_h,
        "fnr_mujeres": fnr_m,
        "disparidad": disparidad,
        "es_sesgado": bool(disparidad > 1.20)
    }

# --- 3. EJEMPLO DE EJECUCIÓN ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_auditar_equidad_genero()
    
    print("--- INPUT (Primeras 5 filas) ---")
    print(entrada["df"].head())
    
    print("\n--- OUTPUT ESPERADO ---")
    for k, v in salida_esperada.items():
        print(f"{k}: {v}")
