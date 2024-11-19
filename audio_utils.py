import numpy as np
import scipy.signal as signal
import noisereduce as nr
import librosa

# Filtro de preénfasis
def filtro_preenfasis(señal, alpha=0.95):  # Ajustado a alpha=0.95 según el script original
    return np.append(señal[0], señal[1:] - alpha * señal[:-1])

# Filtro de paso de banda
def filtro_paso_banda(señal, tasa_muestreo, frec_min=250, frec_max=5500, orden=4):  # Ajustes específicos
    nyquist = 0.5 * tasa_muestreo
    normal_cutoff = [frec_min / nyquist, frec_max / nyquist]
    b, a = signal.butter(orden, normal_cutoff, btype='band')
    return signal.lfilter(b, a, señal)  # Aplicación del filtro

# Reducción de ruido
def eliminar_ruido(señal, tasa_muestreo):
    return nr.reduce_noise(y=señal, sr=tasa_muestreo)

# Normalización de volumen
def normalizar_volumen(señal):
    return señal / np.max(np.abs(señal))  # Normalización completa

# Recorte de silencio
def recortar_audio(señal, tasa_muestreo, umbral_db=20):
    frames = librosa.effects.trim(señal, top_db=umbral_db, frame_length=512, hop_length=64)[0]
    return frames

# Normalización de duración
def normalizar_duracion(señal, tasa_muestreo, duracion_maxima):
    duracion_maxima_muestras = int(tasa_muestreo * duracion_maxima)
    if len(señal) < duracion_maxima_muestras:
        señal_padded = np.pad(señal, (0, duracion_maxima_muestras - len(señal)), 'constant')
    else:
        señal_padded = señal[:duracion_maxima_muestras]
    return señal_padded
