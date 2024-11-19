import os
import csv
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import keyboard  
from AudioProcessor import AudioProcessor
from KnnClassifier import KnnClassifier

# Configuración de grabación
DURACION_MAXIMA = 5  # Duración máxima en segundos (ajusta según tus necesidades)
TASA_MUESTREO = 88200  # Tasa de muestreo en Hz

# Inicializar el procesador de audio y el clasificador KNN
procesador_audio = AudioProcessor()
clasificador_knn = KnnClassifier(k=5)

# Directorio donde se encuentran los archivos de audio de entrenamiento
directorio_audios = './audios'
archivo_csv = 'audio_datos.csv'

# Verificar si el archivo CSV ya existe
if not os.path.exists(archivo_csv):
    print(f"No se encontró el archivo {archivo_csv}, generando uno nuevo...")
    # Crear el archivo CSV para guardar los datos de entrenamiento
    with open(archivo_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Cabecera del CSV: Añadir ZCR, MFCC y Energía para los segmentos
        columnas = ['ZCR_Promedio', 'MFCC_1_Promedio', 'Energia_Promedio', 
                    'MFCC_4_Promedio', 'MFCC_5_Promedio', 'Clase']
        writer.writerow(columnas)
        
        # Procesar todos los archivos de audio en el directorio
        for nombre_archivo in os.listdir(directorio_audios):
            if nombre_archivo.endswith(".wav"):
                ruta_audio = os.path.join(directorio_audios, nombre_archivo)
                
                # Extraer características del archivo de audio
                zcr, mfcc_1, energia, mfcc_4, mfcc_5 = procesador_audio.extraer_caracteristicas_por_segmento(ruta_audio)
                
                # Suponer que el nombre del archivo tiene la clase
                clase = nombre_archivo.split('_')[0]
                
                # Guardar las características y la clase en el archivo CSV
                writer.writerow([zcr, mfcc_1, energia, mfcc_4, mfcc_5, clase])
    print(f"Archivo {archivo_csv} generado con éxito.")
else:
    print(f"El archivo {archivo_csv} ya existe. No es necesario regenerarlo.")

# Entrenar el clasificador KNN con el archivo CSV generado o existente
clasificador_knn.entrenar(archivo_csv)

# Función para grabar audio
def grabar_audio():
    print("Mantén presionada la barra espaciadora para comenzar a grabar. Suéltala para terminar.")
    
    # Esperar hasta que se presione la barra espaciadora para comenzar a grabar
    while not keyboard.is_pressed("space"):
        pass

    grabando = []

    def callback(indata, frames, time, status):
        grabando.extend(indata.copy())

    # Iniciar la grabación mientras se mantenga presionada la barra espaciadora
    with sd.InputStream(samplerate=TASA_MUESTREO, channels=1, callback=callback):
        while keyboard.is_pressed("space"):
            print("Escuchando...", end="\r")  # Imprime "Escuchando..." y sobrescribe en la misma línea
            sd.sleep(100)  # Esperar 100 ms antes de verificar de nuevo

    # Verificar si la grabación contiene audio suficiente (al menos 0.5 segundos)
    min_samples = int(0.5 * TASA_MUESTREO)
    if len(grabando) < min_samples:
        print("Grabación demasiado corta. Inténtalo de nuevo.")
        return None

    # Convertir el audio grabado a un archivo temporal WAV usando soundfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
        sf.write(temp_audio_file.name, np.array(grabando), TASA_MUESTREO)
        return temp_audio_file.name

# Grabar y procesar el audio temporal
ruta_audio_prueba = grabar_audio()
if ruta_audio_prueba:
    print(f"\nArchivo de audio temporal grabado en: {ruta_audio_prueba}")

    # Extraer características del archivo de prueba usando AudioProcessor
    caracteristicas_prueba = procesador_audio.extraer_caracteristicas_por_segmento(ruta_audio_prueba)

    # Mostrar en consola las características extraídas
    print("\nCaracterísticas extraídas del archivo de prueba (promedio de segmentos especificados):")
    print(f"ZCR Promedio: {caracteristicas_prueba[0]:.6f}")
    print(f"MFCC 1 Promedio: {caracteristicas_prueba[1]:.6f}")
    print(f"Energía Promedio: {caracteristicas_prueba[2]:.6f}")
    print(f"MFCC 4 Promedio: {caracteristicas_prueba[3]:.6f}")
    print(f"MFCC 5 Promedio: {caracteristicas_prueba[4]:.6f}")

    # Predecir la clase usando el clasificador Knn
    clase_predicha = clasificador_knn.predecir(caracteristicas_prueba)
    print(f"\nPalabra escuchada: {clase_predicha}")

    # Ruta de la carpeta donde se encuentran las imágenes
    carpeta_imagenes = "Kmeans/Muesta"
    imagen_path = os.path.join(carpeta_imagenes, f"{clase_predicha}.jpg")

    # Mostrar la imagen correspondiente a la clase predicha
    if os.path.exists(imagen_path):
        imagen = cv2.imread(imagen_path)
        if imagen is not None:
            # Usar matplotlib para mostrar la imagen
            from matplotlib import pyplot as plt
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir a RGB para matplotlib
            plt.imshow(imagen_rgb)
            plt.title(f"Imagen: {clase_predicha}")
            plt.axis("off")  # Quitar los ejes
            plt.show()
        else:
            print(f"Error: No se pudo cargar la imagen {imagen_path}")
    else:
        print(f"Imagen no encontrada para la clase predicha: {clase_predicha}")

    # Eliminar el archivo de audio temporal después de la predicción
    os.remove(ruta_audio_prueba)
else:
    print("No se grabó ningún audio válido.")