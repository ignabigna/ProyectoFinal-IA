import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from DatabaseHandler import DatabaseHandler
from sklearn.decomposition import PCA


class KmeansPCA:
    def __init__(self, database_path="database"):
        self.database_path = database_path

    def load_data(self):
        """Carga los datos de los archivos CSV."""
        data = {}
        centroids = []

        for csv_file in os.listdir(self.database_path):
            if csv_file.endswith("_features.csv"):
                vegetable_name = csv_file.split("_")[0]
                csv_path = os.path.join(self.database_path, csv_file)
                data[vegetable_name] = DatabaseHandler.read_csv(csv_path)

            elif csv_file == "clusters.csv":
                csv_path = os.path.join(self.database_path, csv_file)
                centroids = DatabaseHandler.read_csv(csv_path)

        return data, centroids

    def reduce_dimensions(self, features):
        """Reduce las dimensiones de los datos a 2D y 3D utilizando PCA."""
        pca_2d = PCA(n_components=2)
        pca_3d = PCA(n_components=3)

        reduced_2d = pca_2d.fit_transform(features)
        reduced_3d = pca_3d.fit_transform(features)

        return reduced_2d, reduced_3d

    def plot_2d_3d(self, data, centroids):
        """Genera gráficos en 2D y 3D para las nubes de puntos y los clusters."""
        colors = {
            "berenjena": "purple",
            "camote": "orange",
            "papa": "brown",
            "zanahoria": "red",
        }
        features_all = []
        labels_all = []

        # Preparar los datos para PCA
        for vegetable, points in data.items():
            for point in points:
                features_all.append(
                    [
                        float(point["color_R"]),
                        float(point["color_G"]),
                        float(point["color_B"]),
                        float(point["aspect_ratio"]),
                        float(point["roundness"]),
                    ]
                    + [float(point[f"hu_moment_{i + 1}"]) for i in range(7)]
                )
                labels_all.append(vegetable)

        # Agregar los centroides
        for centroid in centroids:
            features_all.append(
                [
                    float(centroid["color_R"]),
                    float(centroid["color_G"]),
                    float(centroid["color_B"]),
                    float(centroid["aspect_ratio"]),
                    float(centroid["roundness"]),
                ]
                + [float(centroid[f"hu_moment_{i + 1}"]) for i in range(7)]
            )
            labels_all.append(f"cluster_{centroid['Vegetable']}")

        # Reducir dimensiones
        reduced_2d, reduced_3d = self.reduce_dimensions(features_all)

        # Graficar 2D
        plt.figure(figsize=(10, 8))
        for vegetable, color in colors.items():
            points_2d = np.array(
                [
                    reduced_2d[i]
                    for i, label in enumerate(labels_all)
                    if label == vegetable
                ]
            )
            plt.scatter(
                points_2d[:, 0],
                points_2d[:, 1],
                c=color,
                label=vegetable,
                alpha=0.7,
            )

        # Agregar centroides al gráfico 2D
        for i, centroid in enumerate(centroids):
            plt.scatter(
                reduced_2d[len(features_all) - len(centroids) + i, 0],
                reduced_2d[len(features_all) - len(centroids) + i, 1],
                c=colors[centroid["Vegetable"]],
                label=f"Cluster {centroid['Vegetable']}",
                edgecolor="black",
                s=150,
                marker="X",
            )

        plt.title("PCA - 2D Projection")
        plt.legend()
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()

        # Graficar 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        for vegetable, color in colors.items():
            points_3d = np.array(
                [
                    reduced_3d[i]
                    for i, label in enumerate(labels_all)
                    if label == vegetable
                ]
            )
            ax.scatter(
                points_3d[:, 0],
                points_3d[:, 1],
                points_3d[:, 2],
                c=color,
                label=vegetable,
                alpha=0.7,
            )

        # Agregar centroides al gráfico 3D
        for i, centroid in enumerate(centroids):
            ax.scatter(
                reduced_3d[len(features_all) - len(centroids) + i, 0],
                reduced_3d[len(features_all) - len(centroids) + i, 1],
                reduced_3d[len(features_all) - len(centroids) + i, 2],
                c=colors[centroid["Vegetable"]],
                label=f"Cluster {centroid['Vegetable']}",
                edgecolor="black",
                s=200,
                marker="X",
            )

        ax.set_title("PCA - 3D Projection")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend()
        plt.show()

    def run(self):
        """Ejecuta el flujo de carga, reducción y graficación."""
        data, centroids = self.load_data()
        self.plot_2d_3d(data, centroids)


if __name__ == "__main__":
    pca_visualizer = KmeansPCA()
    pca_visualizer.run()
