import numpy as np
import csv


class KmeansClassifier:
    def __init__(self, clusters=4, max_iterations=100, tolerance=1e-4):
        """
        Constructor para el algoritmo K-means.
        :param clusters: Número de clusters.
        :param max_iterations: Número máximo de iteraciones.
        :param tolerance: Tolerancia para la convergencia.
        """
        self.clusters = clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None

    def fit(self, features):
        """
        Entrena el modelo K-means.
        :param features: Lista de características (matriz 2D).
        """
        features = np.array(features)
        num_samples, num_features = features.shape

        # Inicialización aleatoria de los centroides
        np.random.seed(42)
        self.centroids = features[np.random.choice(num_samples, self.clusters, replace=False)]

        for i in range(self.max_iterations):
            # Asignar etiquetas según el centroide más cercano
            distances = np.linalg.norm(features[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # Calcular nuevos centroides
            new_centroids = np.array([features[self.labels == j].mean(axis=0) for j in range(self.clusters)])
            
            # Verificar convergencia
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def predict(self, feature):
        """
        Predice el cluster de una nueva característica.
        :param feature: Característica a clasificar (vector 1D).
        :return: Etiqueta del cluster asignado.
        """
        feature = np.array(feature)
        distances = np.linalg.norm(self.centroids - feature, axis=1)
        return np.argmin(distances)

    def save_clusters_to_csv(self, labels, filepath):
        """
        Guarda las etiquetas de los clusters en un archivo CSV.
        :param labels: Etiquetas de los clusters.
        :param filepath: Ruta del archivo CSV de salida.
        """
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Cluster"])
            for i, label in enumerate(labels):
                writer.writerow([f"Image_{i+1}", label])

    def load_clusters_from_csv(self, filepath):
        """
        Carga etiquetas de clusters desde un archivo CSV.
        :param filepath: Ruta del archivo CSV de entrada.
        :return: Etiquetas cargadas como una lista.
        """
        labels = []
        with open(filepath, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar encabezado
            for row in reader:
                labels.append(int(row[1]))
        return labels


# Main para pruebas independientes
if __name__ == "__main__":
    # Prueba con datos simulados
    data = np.random.rand(10, 2)
    classifier = KmeansClassifier()
    classifier.fit(data)
    print(f"Centroids: {classifier.centroids}")
    print(f"Labels: {classifier.labels}")
