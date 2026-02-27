import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# --- 1. FUNCIÓN DE LA MISIÓN: verificar_integridad_senal ---
def verificar_integridad_senal(X_train, X_test, threshold_percentile=95):
    """
    Evalúa la integridad de señales de sensores mediante el error de reconstrucción PCA.
    """
    # Aprendizaje de la Estructura (Sklearn)
    # n_components=0.9 mantiene el 90% de la varianza explicada
    pca = PCA(n_components=0.9, svd_solver='full')
    pca.fit(X_train)
    
    # Compresión y Reconstrucción (Numpy/Sklearn)
    X_test_reduced = pca.transform(X_test)
    X_test_reconstructed = pca.inverse_transform(X_test_reduced)
    
    # Cálculo del Error de Reconstrucción (Numpy)
    # MSE fila por fila entre original y reconstruido
    diferencia = X_test.values - X_test_reconstructed
    mse_fila = np.mean(np.square(diferencia), axis=1)
    
    # Definición de Alerta (Pandas)
    # Umbral dinámico basado en el percentil de los errores
    umbral = np.percentile(mse_fila, threshold_percentile)
    
    # Retorno del DataFrame original con la columna de alerta
    resultado_df = X_test.copy()
    resultado_df['alerta_integridad'] = mse_fila > umbral
    
    return resultado_df

# --- 2. GENERADOR DE CASOS DE USO: generar_caso_de_uso_verificar_integridad_senal ---
def generar_caso_de_uso_verificar_integridad_senal():
    """
    Genera casos de uso aleatorios con dimensiones y ruidos variables.
    """
    # Dimensiones aleatorias para asegurar variabilidad
    n_train = np.random.randint(150, 400)
    n_test = np.random.randint(30, 100)
    n_sensores = 10 # Según especificación del problema
    
    # Generación de estructura de correlación física aleatoria
    # Creamos una señal latente de 2 dimensiones que los sensores deben seguir
    señal_latente_train = np.random.randn(n_train, 2)
    señal_latente_test = np.random.randn(n_test, 2)
    
    # Matriz de mezcla aleatoria (representa cómo cada sensor reacciona a la física)
    mezcla = np.random.uniform(0.5, 3.0, (2, n_sensores))
    
    # Datos normales con ruido blanco
    X_train_raw = señal_latente_train @ mezcla + np.random.normal(0, 0.1, (n_train, n_sensores))
    X_test_raw = señal_latente_test @ mezcla + np.random.normal(0, 0.1, (n_test, n_sensores))
    
    # Inyección de anomalías aleatorias (rompen la correlación)
    n_fallos = np.random.randint(2, 6)
    for _ in range(n_fallos):
        f_idx = np.random.randint(0, n_test)
        c_idx = np.random.randint(0, n_sensores)
        X_test_raw[f_idx, c_idx] += np.random.uniform(10, 20) # Error de escala
        
    # Crear DataFrames
    cols = [f'sensor_{i}' for i in range(n_sensores)]
    X_train = pd.DataFrame(X_train_raw, columns=cols)
    X_test = pd.DataFrame(X_test_raw, columns=cols)
    
    # Parámetros de entrada
    perc = np.random.choice([90, 95, 98])
    input_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "threshold_percentile": perc
    }
    
    # Generar salida esperada
    output_expected = verificar_integridad_senal(X_train, X_test, threshold_percentile=perc)
    
    return input_dict, output_expected

# --- 3. VALIDACIÓN Y SALIDA EN CONSOLA ---
if __name__ == "__main__":
    # Ejecutamos el generador
    inputs, esperado = generar_caso_de_uso_verificar_integridad_senal()
    
    print(f"--- REPORTE DE INTEGRIDAD DE TURBINA ---")
    print(f"Tamaño X_train: {inputs['X_train'].shape}")
    print(f"Tamaño X_test:  {inputs['X_test'].shape}")
    print(f"Percentil usado: {inputs['threshold_percentile']}%")
    print("-" * 40)
    
    # Mostrar filas donde se detectó alerta
    alertas_df = esperado[esperado['alerta_integridad'] == True]
    
    if not alertas_df.empty:
        print(f"Se detectaron {len(alertas_df)} anomalías de integridad:")
        print(alertas_df.head())
    else:
        print("No se detectaron anomalías en este ciclo.")
