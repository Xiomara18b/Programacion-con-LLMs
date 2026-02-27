import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def verificar_integridad_senal(X_train, X_test, threshold_percentile=95):
    """
    Verifica si los datos de sensores mantienen la correlación física
    mediante el error de reconstrucción de PCA.
    """
    # 1. Aprendizaje de la Estructura (90% varianza explicada)
    pca = PCA(n_components=0.9, svd_solver='full')
    pca.fit(X_train)
    
    # 2. Compresión y Reconstrucción
    X_test_compressed = pca.transform(X_test)
    X_test_reconstructed = pca.inverse_transform(X_test_compressed)
    
    # 3. Cálculo del Error de Reconstrucción (MSE fila por fila)
    # Calculamos (original - reconstruido)^2 y promediamos por fila
    mse_per_row = np.mean((X_test.values - X_test_reconstructed)**2, axis=1)
    
    # 4. Definición de Alerta
    # Usamos el percentil de los errores de los mismos datos de prueba (enfoque relativo)
    umbral = np.percentile(mse_per_row, threshold_percentile)
    
    # 5. Retorno
    resultado_df = X_test.copy()
    resultado_df['alerta_integridad'] = mse_per_row > umbral
    
    return resultado_df

def generar_caso_de_uso_verificar_integridad_senal():
    """
    Genera un caso de uso aleatorio (input/output) para la verificación de integridad.
    """
    # Configuraciones aleatorias
    n_train = 200
    n_test = 50
    n_sensores = 10
    percentil = 95
    
    # Crear estructura física base (correlacionada)
    base_train = np.random.randn(n_train, 1)
    # Todos los sensores dependen de la base + un poco de ruido normal
    X_train_data = base_train @ np.ones((1, n_sensores)) + np.random.normal(0, 0.1, (n_train, n_sensores))
    
    base_test = np.random.randn(n_test, 1)
    X_test_data = base_test @ np.ones((1, n_sensores)) + np.random.normal(0, 0.1, (n_test, n_sensores))
    
    # Inyectar anomalía: rompemos la correlación en el sensor 5 de algunas filas
    X_test_data[0:2, 5] += 10.0 # Un sensor descalibrado
    
    # Convertir a DataFrames de Pandas
    cols = [f'sensor_{i}' for i in range(n_sensores)]
    X_train = pd.DataFrame(X_train_data, columns=cols)
    X_test = pd.DataFrame(X_test_data, columns=cols)
    
    # Empaquetar Input
    input_dict = {
        "X_train": X_train,
        "X_test": X_test,
        "threshold_percentile": percentil
    }
    
    # El Output esperado es el resultado de la función lógica
    output_expected = verificar_integridad_senal(X_train, X_test, threshold_percentile=percentil)
    
    return input_dict, output_expected

# --- Ejecución para ver resultados en consola ---
if __name__ == "__main__":
    input_data, output_df = generar_caso_de_uso_verificar_integridad_senal()
    
    print("--- RESULTADOS DE INTEGRIDAD DE SENSORES ---")
    print(output_df.head(10)) # Mostramos las primeras 10 filas
    
    alertas = output_df['alerta_integridad'].sum()
    print(f"\nTotal de lecturas analizadas: {len(output_df)}")
    print(f"Alertas de integridad detectadas: {alertas}")
