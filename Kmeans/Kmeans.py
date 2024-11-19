import os
import numpy as np  # Importación de NumPy
from Processor import Processor
from FeatureExtractor import FeatureExtractor
from KmeansClassifier import KmeansClassifier
from DatabaseHandler import DatabaseHandler
from Comparador import Comparador


class KmeansApp:
    def __init__(self):
        self.processor = Processor()
        self.extractor = FeatureExtractor()
        self.classifier = None
        self.database_path = "database"
        self.vegetables_path = "verduras"

    def generate_database(self):
        """Genera una base de datos procesando imágenes y extrayendo características."""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        print("Generating database...")
        for folder in os.listdir(self.vegetables_path):
            folder_path = os.path.join(self.vegetables_path, folder)
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder}")
                all_features = []

                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    try:
                        # Procesar la imagen
                        processed_image = self.processor.process_image(image_path)

                        # Extraer características
                        features = self.extractor.extract_features(processed_image)
                        feature_row = {
                            "Image": image_name,
                            "color_R": features["color"][0],
                            "color_G": features["color"][1],
                            "color_B": features["color"][2],
                            "aspect_ratio": features["aspect_ratio"],
                            "roundness": features["roundness"],
                        }

                        # Añadir momentos de Hu como columnas separadas
                        for i, moment in enumerate(features["hu_moments"]):
                            feature_row[f"hu_moment_{i + 1}"] = moment

                        all_features.append(feature_row)

                    except ValueError as e:
                        print(f"Error processing {image_path}: {e}")

                # Guardar características en un archivo CSV
                output_csv = os.path.join(self.database_path, f"{folder}_features.csv")
                fieldnames = (
                    ["Image", "color_R", "color_G", "color_B", "aspect_ratio", "roundness"]
                    + [f"hu_moment_{i + 1}" for i in range(7)]
                )
                DatabaseHandler.save_to_csv(all_features, output_csv, fieldnames)
                print(f"Saved features for {folder} to {output_csv}")

    def generate_clusters(self):
        """Genera clusters utilizando el algoritmo K-means para cada archivo CSV de verdura."""
        print("Generating clusters...")
        centroid_data = []
        feature_names = []  # Se determinarán dinámicamente

        # Leer cada archivo CSV de características y calcular un centroide por verdura
        for csv_file in os.listdir(self.database_path):
            if csv_file.endswith("_features.csv"):
                csv_path = os.path.join(self.database_path, csv_file)
                data = DatabaseHandler.read_csv(csv_path)

                # Determinar nombres de columnas dinámicamente (la primera vez)
                if not feature_names:
                    feature_names = [
                        "color_R", "color_G", "color_B", "aspect_ratio", "roundness"
                    ] + [f"hu_moment_{i + 1}" for i in range(7)]

                # Convertir características en una matriz para K-means
                features = [
                    [
                        float(row["color_R"]),
                        float(row["color_G"]),
                        float(row["color_B"]),
                        float(row["aspect_ratio"]),
                        float(row["roundness"]),
                    ] + [float(row[f"hu_moment_{i + 1}"]) for i in range(7)]
                    for row in data
                ]

                # Aplicar K-means con un único cluster al CSV actual
                self.classifier = KmeansClassifier(clusters=1)
                self.classifier.fit(features)

                # Extraer el nombre de la verdura del archivo CSV
                vegetable_name = os.path.splitext(os.path.basename(csv_file))[0].split("_")[0]

                # Guardar el centroide calculado para esta verdura
                centroid = self.classifier.centroids[0]  # Solo un cluster
                centroid_entry = {"Vegetable": vegetable_name}
                centroid_entry.update(
                    {feature_names[j]: centroid[j] for j in range(len(feature_names))}
                )
                centroid_data.append(centroid_entry)

        # Guardar centroides en un CSV
        output_csv = os.path.join(self.database_path, "clusters.csv")
        fieldnames = ["Vegetable"] + feature_names
        DatabaseHandler.save_to_csv(centroid_data, output_csv, fieldnames)
        print(f"Saved centroids to {output_csv}")



    def menu(self):
        """Menú principal de la aplicación."""
        while True:
            print("\nOptions menu:")
            print("1. Generate database")
            print("2. Generate clusters")
            print("3. Classify images")
            print("4. Exit")

            choice = input("Select an option: ").strip()
            if choice == "1":
                self.generate_database()
            elif choice == "2":
                self.generate_clusters()
            elif choice == "3":
                folder_path = input("Enter the path to the folder with images to classify: ").strip()
                comparador = Comparador(self.processor, self.extractor, self.database_path)
                comparador.classify_images(folder_path)
            elif choice == "4":
                print("Exiting the program.")
                break
            else:
                print("Invalid option. Please try again.")



if __name__ == "__main__":
    app = KmeansApp()
    app.menu()
