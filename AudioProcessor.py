import librosa
import numpy as np
from audio_utils import filtro_preenfasis, filtro_paso_banda, eliminar_ruido, normalizar_volumen, recortar_audio

class AudioProcessor:
    def __init__(self, umbral_db=20, umbral_silencio=20, constante=2, operacion="multiplicar"):
        self.umbral_db = umbral_db
        self.umbral_silencio = umbral_silencio  # Umbral de dB para eliminar silencio
        self.constante = constante  # Constante para operaciones adicionales
        self.operacion = operacion  # Operación a realizar ("multiplicar", "elevar")

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

    def dividir_en_segmentos(self, señal, num_segmentos=8):
        longitud_segmento = len(señal) // num_segmentos
        return [señal[i * longitud_segmento:(i + 1) * longitud_segmento] for i in range(num_segmentos)]

    def calcular_zcr(self, segmento):
        zcr = librosa.feature.zero_crossing_rate(segmento)
        return np.sum(zcr)

    def calcular_energia(self, segmento):
        """ Calcula la energía de un segmento como la suma de los cuadrados de las amplitudes """
        return np.sum(np.square(segmento))

    def calcular_mfcc(self, segmento, tasa_muestreo, n_mfcc=6, n_fft=512):
        if len(segmento) < n_fft:
            n_fft = 2 ** (len(segmento) - 1).bit_length()
        mfcc = librosa.feature.mfcc(y=segmento, sr=tasa_muestreo, n_mfcc=n_mfcc, n_fft=n_fft)
        return mfcc

    def aplicar_operacion(self, valores):
        if self.operacion == "multiplicar":
            return [valor * self.constante for valor in valores]
        elif self.operacion == "elevar":
            return [valor ** self.constante for valor in valores]
        return valores

    def extraer_caracteristicas_por_segmento(self, ruta_audio):
        señal_filtrada, tasa_muestreo = self.cargar_y_filtrar_audio(ruta_audio)
        segmentos = self.dividir_en_segmentos(señal_filtrada, num_segmentos=8)

        zcr_totales = []
        energia_totales = []
        mfcc_1_totales = []
        mfcc_4_totales = []
        mfcc_5_totales = []

        # Definir los índices específicos para cada parámetro
        zcr_indices = [0, 2]  # zanahoria
        mfcc_1_indices = [2]  # un poco de todos
        energia_indices = [4, 3]  # camote zanahoria
        mfcc_4_indices = [0]  # un poco todos
        mfcc_5_indices = [3]  #papa

        for i, segmento in enumerate(segmentos):
            if i in zcr_indices:
                zcr = self.calcular_zcr(segmento)
                zcr_totales.append(zcr)

            if i in mfcc_1_indices:
                mfcc = self.calcular_mfcc(segmento, tasa_muestreo)
                mfcc_1_totales.append(mfcc[0])  # MFCC_1

            if i in energia_indices:
                energia = self.calcular_energia(segmento)
                energia_totales.append(energia)  # Energía

            if i in mfcc_4_indices:
                mfcc = self.calcular_mfcc(segmento, tasa_muestreo)
                mfcc_4_totales.append(mfcc[3])  # MFCC_4

            if i in mfcc_5_indices:
                mfcc = self.calcular_mfcc(segmento, tasa_muestreo)
                mfcc_5_totales.append(mfcc[4])  # MFCC_5

        # Aplicar la operación definida a los valores de cada característica
        #zcr_totales = self.aplicar_operacion(zcr_totales)
        #zcr_totales = self.aplicar_operacion(zcr_totales)
        #mfcc_5_totales = self.aplicar_operacion(mfcc_5_totales)
        #mfcc_4_totales = self.aplicar_operacion(mfcc_4_totales)

        # Calcular promedios usando solo los segmentos especificados
        zcr_mediana = np.mean(zcr_totales)
        mfcc_mediana_1 = np.mean(mfcc_1_totales)
        energia_mediana = np.median(energia_totales)
        mfcc_mediana_4 = np.mean(mfcc_4_totales)
        mfcc_mediana_5 = np.mean(mfcc_5_totales)

        # Estandarización (Z-score)
        promedios = np.array([zcr_mediana, mfcc_mediana_1, energia_mediana, mfcc_mediana_4, mfcc_mediana_5])
        media = np.mean(promedios)
        desviacion_estandar = np.std(promedios)

        # Aplicar Z-score
        promedios_estandarizados = (promedios)

        return promedios_estandarizados
