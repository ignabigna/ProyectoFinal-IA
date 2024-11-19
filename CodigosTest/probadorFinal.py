import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import tempfile
from audio_utils import (filtro_preenfasis, 
                         filtro_paso_banda, 
                         eliminar_ruido, 
                         normalizar_volumen, 
                         recortar_audio)

# Inicializar el procesador de audio
class AudioProcessor:
    def __init__(self, umbral_db=15, umbral_silencio=20):
        self.umbral_db = umbral_db
        self.umbral_silencio = umbral_silencio  # Umbral de dB para eliminar silencio

    def cargar_y_filtrar_audio(self, ruta_audio):
        tasa_muestreo_estandar = 88200  # Ajuste de tasa de muestreo según el script original
        señal, tasa_muestreo = librosa.load(ruta_audio, sr=tasa_muestreo_estandar)

        # Aplicar los filtros de preénfasis, paso de banda y reducción de ruido
        señal = filtro_preenfasis(señal)
        señal = filtro_paso_banda(señal, tasa_muestreo)
        señal = eliminar_ruido(señal, tasa_muestreo)

        # Eliminar segmentos de silencio en la señal
        señal = self.eliminar_silencio(señal, tasa_muestreo)

        # Normalizar volumen
        señal_normalizada = normalizar_volumen(señal)

        return señal_normalizada, tasa_muestreo

    def eliminar_silencio(self, señal, tasa_muestreo):
        # Identificar intervalos no silenciosos
        intervalos_no_silenciosos = librosa.effects.split(señal, top_db=self.umbral_silencio)
        
        # Concatenar las partes no silenciosas
        señal_filtrada = np.concatenate([señal[inicio:fin] for inicio, fin in intervalos_no_silenciosos])
        
        return señal_filtrada

# Procesar y graficar los audios
def procesar_audio(ruta_audio_entrada):
    procesador_audio = AudioProcessor()  # Inicializa el procesador de audio
    señal_procesada, tasa_muestreo = procesador_audio.cargar_y_filtrar_audio(ruta_audio_entrada)

    # Crear un archivo temporal para el audio procesado
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        sf.write(temp_audio_file.name, señal_procesada, tasa_muestreo)  # Guardar audio en el archivo temporal
        ruta_temporal = temp_audio_file.name

    return señal_procesada, tasa_muestreo, ruta_audio_entrada, ruta_temporal  # Retorna la señal procesada, la ruta del archivo original y la ruta temporal

# Procesar todos los archivos de audio en el directorio
directorio_audios = './audios'
for nombre_archivo in os.listdir(directorio_audios):
    if nombre_archivo.endswith(".wav"):  # Solo procesar archivos .wav
        ruta_audio = os.path.join(directorio_audios, nombre_archivo)
        print(f"Procesando archivo: {ruta_audio}")
        señal_procesada, tasa, ruta_audio_original, ruta_temporal = procesar_audio(ruta_audio)

        # Graficar audio original
        plt.figure(figsize=(12, 6))

        # Gráfico del audio original
        señal_original, _ = librosa.load(ruta_audio_original, sr=None)  # Cargar el audio original sin cambiar la tasa de muestreo
        tiempo_original = np.arange(len(señal_original)) / tasa
        plt.subplot(2, 1, 1)
        plt.plot(tiempo_original, señal_original, color='blue')
        plt.title(f'Espectro del Audio Original - {nombre_archivo}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.grid()

        # Gráfico del audio procesado
        tiempo_procesado = np.arange(len(señal_procesada)) / tasa
        plt.subplot(2, 1, 2)
        plt.plot(tiempo_procesado, señal_procesada, color='green')
        plt.title(f'Espectro del Audio Procesado - {nombre_archivo}')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.grid()

        plt.tight_layout()
        plt.show()

        # Eliminar el archivo temporal después de mostrar el gráfico
        os.remove(ruta_temporal)
