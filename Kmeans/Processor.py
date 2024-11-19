import cv2
import numpy as np
import os


class Processor:
    def __init__(self, base_path="verduras"):
        """
        Constructor de la clase Processor.
        :param base_path: Ruta base donde se encuentran las carpetas de las verduras.
        """
        self.base_path = base_path

    def increase_saturation(self, image, factor=1.9):
        """Aumenta la saturación de la imagen."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def increase_exposure(self, image, factor=0.35):
        """Aumenta la exposición de la imagen."""
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel = np.clip(l_channel * (1 + factor), 0, 255).astype(np.uint8)
        enhanced_image = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    def apply_smoothing_filter(self, image, kernel_size=5):
        """Aplica un filtro de suavizado."""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def remove_light_background(self, image):
        """Elimina el fondo claro de la imagen."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        object_mask = cv2.bitwise_not(background_mask)
        return cv2.bitwise_and(image, image, mask=object_mask)

    def paint_shape_with_average_color(self, image, background_removed):
        """Pinta la forma recortada con el color promedio."""
        gray_image = cv2.cvtColor(background_removed, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary_image)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        non_black_pixels = image[mask == 255]
        average_color = np.mean(non_black_pixels, axis=0).astype(np.uint8)

        colored_shape = np.zeros_like(image)
        colored_shape[mask == 255] = average_color

        return colored_shape

    def process_image(self, image_path):
        """Procesa una imagen individual."""
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image from path: {image_path}")

        # Aumentar saturación
        increased_saturation_image = self.increase_saturation(original_image)

        # Aumentar exposición
        increased_exposure_image = self.increase_exposure(increased_saturation_image)

        # Aplicar filtro de suavizado
        smoothed_image = self.apply_smoothing_filter(increased_exposure_image)

        # Eliminar fondo claro
        background_removed_image = self.remove_light_background(smoothed_image)

        # Pintar la forma recortada con el color promedio
        colored_shape_image = self.paint_shape_with_average_color(smoothed_image, background_removed_image)

        return colored_shape_image

    def process_folder(self):
        """
        Procesa todas las imágenes dentro de las subcarpetas de la carpeta base.
        """
        for folder in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder)
            if os.path.isdir(folder_path):
                print(f"Processing folder: {folder}")
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    try:
                        processed_image = self.process_image(image_path)
                        # Aquí puedes guardar la imagen procesada si es necesario
                        # cv2.imwrite(f"processed/{folder}_{image_name}", processed_image)
                    except ValueError as e:
                        print(f"Error processing {image_path}: {e}")


# Main para pruebas independientes
if __name__ == "__main__":
    processor = Processor()
    processor.process_folder()
