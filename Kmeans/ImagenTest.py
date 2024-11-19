import cv2
import matplotlib.pyplot as plt
import numpy as np
from Processor import Processor


class ImagenTest:
    def __init__(self):
        self.preprocessor = Processor()

    def reduce_contrast(self, image, factor=0.5):
        """Reduce el contraste de la imagen en un porcentaje determinado."""
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Reducir el rango de valores del canal L
        l_channel = cv2.normalize(l_channel, None, 0, int(255 * factor), cv2.NORM_MINMAX)

        reduced_contrast_image = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(reduced_contrast_image, cv2.COLOR_LAB2BGR)    

    def increase_saturation(self, image, factor=1.9):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * factor, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def increase_exposure(self, image, factor=0.35):
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab_image)
        l_channel = np.clip(l_channel * (1 + factor), 0, 255).astype(np.uint8)
        enhanced_image = cv2.merge((l_channel, a_channel, b_channel))
        return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

    def apply_smoothing_filter(self, image, kernel_size=5):
        smoothed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return smoothed_image

    def remove_light_background(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, background_mask = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
        object_mask = cv2.bitwise_not(background_mask)
        result = cv2.bitwise_and(image, image, mask=object_mask)
        return result

    def paint_shape_with_average_color(self, image, background_removed):
        gray_image = cv2.cvtColor(background_removed, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary_image)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Calcular el color promedio de la región no negra en "background_removed"
        non_black_pixels = background_removed[mask == 255]
        average_color = np.mean(non_black_pixels, axis=0).astype(np.uint8)

        # Crear una nueva imagen con el color promedio aplicado a la máscara
        colored_shape = np.zeros_like(background_removed)
        colored_shape[mask == 255] = average_color

        return colored_shape

    def display_image_processing(self, image_path):
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Error: Could not load image from path '{image_path}'. Check the file path.")

        # Aumentar saturación
        increased_saturation_image = self.increase_saturation(original_image)

        # Aumentar exposición
        increased_exposure_image = self.increase_exposure(increased_saturation_image, factor=0.6)

        # Aplicar filtro de suavizado
        smoothed_image = self.apply_smoothing_filter(increased_exposure_image)

        # Eliminar fondo claro
        background_removed_image = self.remove_light_background(smoothed_image)

        # Pintar la forma recortada con el color promedio
        colored_shape_image = self.paint_shape_with_average_color(smoothed_image, background_removed_image)

        # Mostrar las imágenes procesadas
        fig, axs = plt.subplots(1, 5, figsize=(20, 5))

        axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(cv2.cvtColor(increased_saturation_image, cv2.COLOR_BGR2RGB))
        axs[1].set_title("Increased Saturation")
        axs[1].axis('off')

        axs[2].imshow(cv2.cvtColor(increased_exposure_image, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Increased Exposure")
        axs[2].axis('off')

        axs[3].imshow(cv2.cvtColor(background_removed_image, cv2.COLOR_BGR2RGB))
        axs[3].set_title("Background Removed")
        axs[3].axis('off')

        axs[4].imshow(cv2.cvtColor(colored_shape_image, cv2.COLOR_BGR2RGB))
        axs[4].set_title("Average Color Shape")
        axs[4].axis('off')

        plt.tight_layout()
        plt.show()


# Main para pruebas independientes
if __name__ == "__main__":
    image_path = input("Enter the path to an image: ")
    tester = ImagenTest()
    try:
        tester.display_image_processing(image_path)
    except ValueError as e:
        print(f"Error: {e}")
