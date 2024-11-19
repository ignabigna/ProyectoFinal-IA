import librosa
import numpy as np
import pandas as pd
import os
from audio_utils import filtro_preenfasis, filtro_paso_banda, eliminar_ruido, normalizar_volumen

class AudioProcessor:
    def __init__(self, umbral_silencio=20):
        self.umbral_silencio = umbral_silencio  # Umbral de dB para eliminar silencio

    def cargar_y_filtrar_audio(self, ruta_audio):
        tasa_muestreo_estandar = 88200  # Configuración de tasa de muestreo
        señal, tasa_muestreo = librosa.load(ruta_audio, sr=tasa_muestreo_estandar)

        # Aplicar filtros de preénfasis, pasa-banda, y eliminación de ruido
        señal = filtro_preenfasis(señal)
        señal = filtro_paso_banda(señal, tasa_muestreo)
        señal = eliminar_ruido(señal, tasa_muestreo)

        # Eliminar segmentos de silencio
        señal = self.eliminar_silencio(señal, tasa_muestreo)

        # Normalizar volumen
        señal_normalizada = normalizar_volumen(señal)

        return señal_normalizada, tasa_muestreo

    def eliminar_silencio(self, señal, tasa_muestreo):
        intervalos_no_silenciosos = librosa.effects.split(señal, top_db=self.umbral_silencio)
        señal_filtrada = np.concatenate([señal[inicio:fin] for inicio, fin in intervalos_no_silenciosos])
        return señal_filtrada

    def dividir_en_segmentos(self, señal, num_segmentos=8):
        """ Divide la señal en segmentos iguales """
        longitud_segmento = len(señal) // num_segmentos
        return [señal[i * longitud_segmento:(i + 1) * longitud_segmento] for i in range(num_segmentos)]

    def calcular_zcr(self, segmento):
        """ Calcula el ZCR promedio de un segmento """
        zcr = librosa.feature.zero_crossing_rate(segmento)
        return np.sum(zcr)

    def calcular_amplitud(self, segmento):
        """ Calcula la amplitud media de un segmento """
        return np.mean(np.abs(segmento))

    def calcular_energia(self, segmento):
        """ Calcula la energía de un segmento como la suma de los cuadrados de las amplitudes """
        return np.sum(np.square(segmento))

    def calcular_mfcc(self, segmento, tasa_muestreo, n_mfcc=6, n_fft=512):
        """ Calcula los MFCC para los coeficientes 1, 4 y 5 de un segmento y devuelve el promedio de cada uno """
        mfcc = librosa.feature.mfcc(y=segmento, sr=tasa_muestreo, n_mfcc=n_mfcc, n_fft=n_fft)
        return np.mean(mfcc[0]), np.mean(mfcc[3]), np.mean(mfcc[4])

    def extraer_caracteristicas_por_segmento(self, ruta_audio):
        """ Divide el audio en segmentos y calcula las métricas para cada segmento """
        señal_filtrada, tasa_muestreo = self.cargar_y_filtrar_audio(ruta_audio)
        segmentos = self.dividir_en_segmentos(señal_filtrada, num_segmentos=8)

        # Inicializar listas para almacenar resultados de cada segmento
        zcr_totales = []
        amplitud_totales = []
        energia_totales = []
        mfcc_1_totales = []
        mfcc_4_totales = []
        mfcc_5_totales = []

        for segmento in segmentos:
            zcr_totales.append(self.calcular_zcr(segmento))
            amplitud_totales.append(self.calcular_amplitud(segmento))
            energia_totales.append(self.calcular_energia(segmento))
            
            mfcc_1, mfcc_4, mfcc_5 = self.calcular_mfcc(segmento, tasa_muestreo)
            mfcc_1_totales.append(mfcc_1)
            mfcc_4_totales.append(mfcc_4)
            mfcc_5_totales.append(mfcc_5)

        return zcr_totales, amplitud_totales, energia_totales, mfcc_1_totales, mfcc_4_totales, mfcc_5_totales

# Ruta de la carpeta que contiene los archivos de audio
carpeta_audios = 'audios/'
procesador = AudioProcessor()

# Inicializar diccionarios para almacenar los resultados de todos los archivos
zcr_data = {}
amplitud_data = {}
energia_data = {}
mfcc_1_data = {}
mfcc_4_data = {}
mfcc_5_data = {}

# Procesar todos los archivos de audio en la carpeta
for archivo in os.listdir(carpeta_audios):
    if archivo.endswith(".wav"):  # Asegúrate de que sea un archivo de audio
        ruta_audio = os.path.join(carpeta_audios, archivo)
        nombre_audio = os.path.splitext(archivo)[0]  # Nombre base del archivo sin extensión

        # Extraer características por segmento
        zcr, amplitud, energia, mfcc_1, mfcc_4, mfcc_5 = procesador.extraer_caracteristicas_por_segmento(ruta_audio)
        
        # Almacenar los resultados en el diccionario correspondiente
        zcr_data[nombre_audio] = zcr
        amplitud_data[nombre_audio] = amplitud
        energia_data[nombre_audio] = energia
        mfcc_1_data[nombre_audio] = mfcc_1
        mfcc_4_data[nombre_audio] = mfcc_4
        mfcc_5_data[nombre_audio] = mfcc_5

# Crear y guardar cada DataFrame con los resultados de todos los archivos
pd.DataFrame.from_dict(zcr_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("zcr_segmentos.csv")
pd.DataFrame.from_dict(amplitud_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("amplitud_segmentos.csv")
pd.DataFrame.from_dict(energia_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("energia_segmentos.csv")
pd.DataFrame.from_dict(mfcc_1_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("mfcc_1_segmentos.csv")
pd.DataFrame.from_dict(mfcc_4_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("mfcc_4_segmentos.csv")
pd.DataFrame.from_dict(mfcc_5_data, orient='index', columns=[f"Segmento_{i+1}" for i in range(8)]).to_csv("mfcc_5_segmentos.csv")

print("Archivos CSV generados con éxito.")
