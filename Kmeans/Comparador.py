import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Processor import Processor
from FeatureExtractor import FeatureExtractor
from DatabaseHandler import DatabaseHandler


class Comparador:
    def __init__(self, processor, extractor, database_path):
        """
        Inicializa la clase Comparador.
        :param processor: Instancia de la clase Processor para procesar imágenes.
        :param extractor: Instancia de la clase FeatureExtractor para extraer características.
        :param database_path: Ruta donde se encuentra el archivo de clusters.
        """
        self.processor = processor
        self.extractor = extractor
        self.database_path = database_path

    def load_clusters(self):
        """
        Carga los centroides de los clusters desde el archivo CSV.
        :return: Lista de clusters con sus parámetros.
        """
        clusters_file = os.path.join(self.database_path, "clusters.csv")
        if not os.path.exists(clusters_file):
            raise FileNotFoundError("Clusters file not found. Generate clusters first.")
        return DatabaseHandler.read_csv(clusters_file)

    def classify_images(self, folder_path):
        """
        Clasifica las imágenes en la carpeta proporcionada, guarda una copia con el nombre de la verdura y muestra las imágenes clasificadas.
        :param folder_path: Ruta de la carpeta con imágenes.
        """
        if not os.path.exists(folder_path):
            print("Error: Folder not found.")
            return

        try:
            clusters = self.load_clusters()
        except FileNotFoundError as e:
            print(e)
            return

        images_with_labels = []

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            try:
                # Procesar y extraer características
                processed_image = self.processor.process_image(image_path)
                features = self.extractor.extract_features(processed_image)
                feature_vector = [
                    features["color"][0],
                    features["color"][1],
                    features["color"][2],
                    features["aspect_ratio"],
                    features["roundness"],
                ] + features["hu_moments"]

                print(f"\nImage: {image_name}")
                print("Extracted parameters:")
                print(f"Color_R: {features['color'][0]:.5f}")
                print(f"Color_G: {features['color'][1]:.5f}")
                print(f"Color_B: {features['color'][2]:.5f}")
                print(f"Aspect Ratio: {features['aspect_ratio']:.5f}")
                print(f"Roundness: {features['roundness']:.5f}")
                for i, hu_moment in enumerate(features["hu_moments"], start=1):
                    print(f"Hu_Moment_{i}: {hu_moment:.5f}")

                # Clasificar imagen basándose en distancia euclidiana
                min_distance = float("inf")
                closest_cluster = "Unknown"

                for cluster in clusters:
                    cluster_vector = [
                        float(cluster["color_R"]),
                        float(cluster["color_G"]),
                        float(cluster["color_B"]),
                        float(cluster["aspect_ratio"]),
                        float(cluster["roundness"]),
                    ] + [float(cluster[f"hu_moment_{i + 1}"]) for i in range(7)]

                    # Calcular distancia euclidiana
                    distance = np.linalg.norm(np.array(feature_vector) - np.array(cluster_vector))
                    if distance < min_distance:
                        min_distance = distance
                        closest_cluster = cluster["Vegetable"]

                print(f"\nClosest Cluster: {closest_cluster} (Euclidean Distance: {min_distance:.5f})")

                # Guardar copia con el nombre de la verdura y redimensionar a 600x600
                new_image_name = f"{closest_cluster}.jpg"
                new_image_path = os.path.join(folder_path, new_image_name)
                original_image = cv2.imread(image_path)

                if original_image is not None:
                    resized_image = cv2.resize(original_image, (600, 600))
                    cv2.imwrite(new_image_path, resized_image)
                    images_with_labels.append((resized_image, new_image_name))
                    print(f"Saved a resized copy of the image as: {new_image_path}")

            except Exception as e:
                print(f"Error en procesamiento de {image_name}: {e}")

        # Mostrar las imágenes clasificadas con nombres
        if images_with_labels:
            self.display_images_with_labels(images_with_labels)

    def display_images_with_labels(self, images_with_labels):
        """
        Muestra las imágenes clasificadas con sus nombres en un mosaico usando matplotlib.
        :param images_with_labels: Lista de tuplas (imagen, nombre de archivo).
        """
        num_images = len(images_with_labels)
        cols = min(4, num_images)  # Máximo 4 columnas en el mosaico
        rows = (num_images + cols - 1) // cols  # Número de filas necesarias

        # Crear subplots
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

        # Asegurarnos de que axes sea una lista plana
        if rows == 1 and cols == 1:  # Caso especial: una sola imagen
            axes = [axes]
        elif rows == 1 or cols == 1:  # Caso especial: una sola fila o columna
            axes = axes.flatten()
        else:
            axes = axes.ravel()

        # Procesar cada imagen
        for i, (image, label) in enumerate(images_with_labels):
            # Convertir imagen a RGB para matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            ax = axes[i]
            ax.imshow(image_rgb)
            ax.set_title(label, fontsize=12)
            ax.axis('off')

        # Ocultar celdas no usadas
        for j in range(len(images_with_labels), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()
