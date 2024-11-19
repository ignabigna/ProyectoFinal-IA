import csv


class DatabaseHandler:
    @staticmethod
    def read_csv(filepath):
        """
        Lee un archivo CSV y devuelve sus datos como una lista de diccionarios.
        :param filepath: Ruta del archivo CSV.
        :return: Lista de diccionarios.
        """
        data = []
        with open(filepath, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    @staticmethod
    def save_to_csv(data, filepath, fieldnames):
        """
        Guarda datos en un archivo CSV.
        :param data: Lista de diccionarios con los datos a guardar.
        :param filepath: Ruta del archivo CSV de salida.
        :param fieldnames: Lista de nombres de columnas.
        """
        with open(filepath, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


# Main para pruebas independientes
if __name__ == "__main__":
    # Prueba de guardar y leer CSV
    test_data = [
        {"Image": "papa_1", "Aspect_Ratio": 1.5, "Roundness": 0.8, "Cluster": 2},
        {"Image": "zanahoria_1", "Aspect_Ratio": 3.2, "Roundness": 0.7, "Cluster": 1},
    ]
    filepath = "test_database.csv"

    # Guardar datos
    DatabaseHandler.save_to_csv(test_data, filepath, fieldnames=["Image", "Aspect_Ratio", "Roundness", "Cluster"])

    # Leer datos
    loaded_data = DatabaseHandler.read_csv(filepath)
    print("Loaded Data:", loaded_data)
