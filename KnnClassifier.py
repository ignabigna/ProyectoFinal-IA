import csv
import numpy as np
from collections import Counter

class KnnClassifier:
    def __init__(self, k=5):
        self.k = k
        self.datos_entrenamiento = []
        self.clases_entrenamiento = []

    def entrenar(self, archivo_csv):
        """Lee los datos desde el archivo CSV y los guarda en el clasificador."""
        with open(archivo_csv, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la cabecera
            for row in reader:
                caracteristicas = np.array([float(valor) for valor in row[:-1]])  # Convertir a float
                clase = row[-1]
                self.datos_entrenamiento.append(caracteristicas)
                self.clases_entrenamiento.append(clase)

    def distancia_euclidiana(self, v1, v2):
        """Calcula la distancia euclidiana entre dos vectores."""
        return np.sqrt(np.sum((v1 - v2)**2))

    def predecir(self, dato_prueba):
        """Predice la clase basándose en los 5 vecinos más cercanos."""
        distancias = []

        # Calcular la distancia entre el dato de prueba y todos los datos de entrenamiento
        for i in range(len(self.datos_entrenamiento)):
            dist = self.distancia_euclidiana(self.datos_entrenamiento[i], dato_prueba)
            distancias.append((dist, self.clases_entrenamiento[i]))

        # Ordenar las distancias y obtener los vecinos más cercanos
        distancias.sort(key=lambda x: x[0])
        
        # Mostrar los 5 vecinos más cercanos
        print(f"\nVecinos más cercanos:")
        for i in range(self.k):
            print(f"Vecino {i+1}: Clase = {distancias[i][1]}, Distancia = {distancias[i][0]:.6f}")

        # Realizar un voto mayoritario basado en los k vecinos más cercanos
        clases_votacion = [distancias[i][1] for i in range(self.k)]
        clase_predicha = Counter(clases_votacion).most_common(1)[0][0]

        return clase_predicha
