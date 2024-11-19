import os
import librosa
import numpy as np
import pandas as pd
from audio_utils import filtro_preenfasis, filtro_paso_banda, eliminar_ruido, normalizar_volumen, recortar_audio

class EnergySegmentProcessor:
    def __init__(self, umbral_silencio=20):
        self.umbral_silencio = umbral_silencio

    def cargar_y_filtrar_audio(self, ruta_audio):
        tasa_muestreo_estandar = 88200
        señal, tasa_muestreo = librosa.load(ruta_audio, sr=tasa_muestreo_estandar)

        # Aplicar filtros
        señal = filtro_preenfasis(señal)
        señal = filtro_paso_banda(señal, tasa_muestreo)
        señal = eliminar_ruido(señal, tasa_muestreo)

        # Eliminar segmentos de silencio
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

    def dividir_en_segmentos(self, señal, num_segmentos=8):
        longitud_segmento = len(señal) // num_segmentos
        return [señal[i * longitud_segmento:(i + 1) * longitud_segmento] for i in range(num_segmentos)]

    def calcular_energia(self, segmento):
        # Calcular la energía del segmento como la suma de los cuadrados de sus amplitudes
        energia = np.sum(segmento ** 2)
        return energia

    def procesar_audio(self, ruta_audio):
        señal_filtrada, tasa_muestreo = self.cargar_y_filtrar_audio(ruta_audio)
        segmentos = self.dividir_en_segmentos(señal_filtrada, num_segmentos=8)

        energia_valores = []
        for segmento in segmentos:
            energia_promedio = self.calcular_energia(segmento)
            energia_valores.append(energia_promedio)

        return energia_valores

    def guardar_en_csv(self, ruta_archivo_csv, audio_nombre, energia_valores):
        # Asegúrate de que haya 10 valores en la lista
        while len(energia_valores) < 8:
            energia_valores.append(np.nan)  # Rellenar con NaN si hay menos de 10

        # Crear un DataFrame con la energía de cada segmento
        df = pd.DataFrame([energia_valores], columns=[f'segmento{i+1}' for i in range(len(energia_valores))])
        df.insert(0, 'audio', audio_nombre)  # Agregar nombre de audio
        df.to_csv(ruta_archivo_csv, mode='a', index=False, header=not os.path.exists(ruta_archivo_csv))

# Ejemplo de uso:
if __name__ == "__main__":
    procesador_energia = EnergySegmentProcessor()
    directorio_audios = './audios'  # Cambia esto según tu estructura de directorios
    archivo_csv = 'energia_segmentos.csv'

    # Procesar todos los archivos de audio en el directorio
    for nombre_archivo in os.listdir(directorio_audios):
        if nombre_archivo.endswith(".wav"):
            ruta_audio = os.path.join(directorio_audios, nombre_archivo)

            # Procesar el audio y obtener la energía de cada segmento
            energia_valores = procesador_energia.procesar_audio(ruta_audio)

            # Guardar en CSV
            procesador_energia.guardar_en_csv(archivo_csv, nombre_archivo, energia_valores)

    print(f"Archivo {archivo_csv} generado con éxito.")
