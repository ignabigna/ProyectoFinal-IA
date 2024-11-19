import pandas as pd
import numpy as np

# Cargar el archivo CSV
file_path = 'audio_datos.csv'
df = pd.read_csv(file_path)

# Definir los límites de la normalización
min_target, max_target = -100, 100

# Seleccionar las columnas numéricas (excluyendo la columna 'Clase')
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Función de normalización
def normalize_column(column):
    min_col, max_col = column.min(), column.max()
    # Evitar división por cero si todos los valores son iguales
    if min_col == max_col:
        return np.full(column.shape, (min_target + max_target) / 2)
    # Normalizar al rango -20 y 20
    return min_target + (column - min_col) * (max_target - min_target) / (max_col - min_col)

# Aplicar la normalización a cada columna numérica
df[numeric_columns] = df[numeric_columns].apply(normalize_column)

# Guardar el archivo sobrescribiendo el original
df.to_csv(file_path, index=False)
print("Archivo normalizado y guardado.")
