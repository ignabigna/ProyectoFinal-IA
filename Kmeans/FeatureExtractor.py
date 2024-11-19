import cv2
import numpy as np


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_color_features(self, image):
        """Extrae el color promedio ignorando el fondo blanco."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        mask = binary_image == 255
        non_white_pixels = image[mask]
        if len(non_white_pixels) == 0:
            raise ValueError("No valid region found to extract color features.")
        average_color = np.mean(non_white_pixels, axis=0).astype(np.uint8)
        return average_color

    def convert_to_grayscale(self, image):
        """Convierte la imagen a escala de grises."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def binarize_image(self, grayscale_image):
        """Binariza la imagen usando Otsu."""
        _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_image

    def extract_aspect_ratio(self, contour):
        """Calcula la relación de aspecto de un contorno."""
        x, y, w, h = cv2.boundingRect(contour)
        return w / h if h != 0 else 0

    def extract_roundness(self, contour):
        """Calcula la redondez de un contorno."""
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter ** 2)

    def extract_hu_moments(self, contour):
        """Calcula los momentos de Hu del contorno."""
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        return hu_moments

    def extract_features(self, image):
        """Extrae todas las características de una imagen."""
        # Escala de grises y binarización
        grayscale_image = self.convert_to_grayscale(image)
        binary_image = self.binarize_image(grayscale_image)

        # Encontrar contornos
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image.")

        # Extraer características del contorno más grande
        largest_contour = max(contours, key=cv2.contourArea)
        aspect_ratio = self.extract_aspect_ratio(largest_contour)
        roundness = self.extract_roundness(largest_contour)
        hu_moments = self.extract_hu_moments(largest_contour)

        # Extraer color promedio
        color_features = self.extract_color_features(image)

        # Combinar características
        features = {
            "aspect_ratio": aspect_ratio,
            "roundness": roundness,
            "hu_moments": hu_moments.tolist(),
            "color": color_features.tolist()
        }
        return features


# Main para pruebas independientes
if __name__ == "__main__":
    import os

    # Ruta de prueba
    test_image_path = os.path.join("verduras", "papa", "papa_1.jpg")
    image = cv2.imread(test_image_path)

    if image is None:
        print(f"Could not load image: {test_image_path}")
    else:
        extractor = FeatureExtractor()
        try:
            features = extractor.extract_features(image)
            print(f"Extracted features: {features}")
        except ValueError as e:
            print(f"Error: {e}")
