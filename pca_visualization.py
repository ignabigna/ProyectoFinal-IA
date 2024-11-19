import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cargar_datos_csv(archivo_csv):
    """Carga los datos del archivo CSV y separa las características de las clases."""
    datos = pd.read_csv(archivo_csv)
    
    # Separar las características (todas las columnas excepto 'Clase') y las clases
    X = datos.drop('Clase', axis=1)
    y = datos['Clase']
    return X, y, datos

def aplicar_pca(X, n_componentes=3):
    """Aplica PCA a las características y devuelve los datos transformados."""
    # Escalar los datos antes de aplicar PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA
    pca = PCA(n_components=n_componentes)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca

def graficar_pca(X_pca, y):
    """Genera un gráfico de los datos PCA en 2D y 3D."""
    # Gráfico 2D
    plt.figure(figsize=(8, 6))
    
    # Convertir las clases a un array de colores únicos para el gráfico
    clases_unicas = np.unique(y)
    colores = plt.cm.get_cmap('tab10', len(clases_unicas))

    for i, clase in enumerate(clases_unicas):
        plt.scatter(X_pca[y == clase, 0], X_pca[y == clase, 1], label=clase, color=colores(i))

    plt.title("Gráfico PCA de las características del CSV (2D)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title='Clase')
    plt.grid(True)

    # Mostrar información de cada punto en la consola
    for i in range(X_pca.shape[0]):
        print(f"Fila {i}: CP 1 = {X_pca[i, 0]:.2f}, CP 2 = {X_pca[i, 1]:.2f}, Clase = {y.iloc[i]}")

    plt.show()

    # Gráfico 3D
    if X_pca.shape[1] >= 3:  # Verificar si hay al menos 3 componentes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, clase in enumerate(clases_unicas):
            indices = y == clase
            ax.scatter(X_pca[indices, 0], X_pca[indices, 1], X_pca[indices, 2], label=clase, color=colores(i))
        
        ax.set_title("Gráfico PCA de las características del CSV (3D)")
        ax.set_xlabel("Componente Principal 1")
        ax.set_ylabel("Componente Principal 2")
        ax.set_zlabel("Componente Principal 3")
        ax.legend(title='Clase')
        ax.grid(True)

        # Mostrar información de cada punto en la consola
        for i in range(X_pca.shape[0]):
            print(f"Fila {i}: CP 1 = {X_pca[i, 0]:.2f}, CP 2 = {X_pca[i, 1]:.2f}, CP 3 = {X_pca[i, 2]:.2f}, Clase = {y.iloc[i]}")

        plt.show()

if __name__ == "__main__":
    # Ruta al archivo CSV generado
    archivo_csv = 'audio_datos.csv'
    
    # Cargar los datos y las clases
    X, y, datos = cargar_datos_csv(archivo_csv)
    
    # Aplicar PCA
    X_pca = aplicar_pca(X, n_componentes=3)  # Cambiado a 3 componentes para el gráfico 3D
    
    # Generar los gráficos de PCA
    graficar_pca(X_pca, y)
