import pyaudio
import numpy as np
import librosa 
import threading
from scipy import signal
import time

mic_active = False
audio_data = [] 

def calculate_frequency(audio_data, sample_rate):
    """
    Calcula la frecuencia dominante en el audio utilizando la transformada de Fourier.
    """
    # Aplica la transformada de Fourier
    fft_data = np.fft.fft(audio_data)
    # Calcula la magnitud de las frecuencias
    magnitude = np.abs(fft_data)
    # Encuentra el índice de la frecuencia dominante
    dominant_index = np.argmax(magnitude)
    # Calcula la frecuencia correspondiente al índice
    frequency = dominant_index * sample_rate / len(audio_data)
    return frequency

def record_audio():
    global mic_active, audio_data
    duration = 5  # Duración de la grabación en segundos
    sample_rate = 44100  # Tasa de muestreo en Hz
    chunk = 1024  # Tamaño del búfer de audio
    # Coeficientes del filtro de preénfasis y post-enfasis
    preemphasis_coeff = 0.97
    postemphasis_coeff = 1.0 / preemphasis_coeff
    preemphasis_filter = [1, -preemphasis_coeff]
    postemphasis_filter = [1, -postemphasis_coeff]
    
    # Inicia la grabación
    audio_data = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sample_rate, input=True, frames_per_buffer=chunk)
    mic_active = True
    print("Grabando audio... (Presiona '2' para detener)")
    for _ in range(int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        # Aplica el filtro de preénfasis
        data = signal.lfilter(preemphasis_filter, 1, np.frombuffer(data, dtype=np.float32))
        audio_data.append(data)
        if not mic_active:
            break
    stream.stop_stream()
    stream.close()
    p.terminate()
    mic_active = False
    print("Grabación finalizada.")
    
    # Concatena los fragmentos de audio en un solo array
    audio_data = np.concatenate(audio_data)
    
    # Aplica el filtro de post-enfasis
    audio_data = signal.lfilter(postemphasis_filter, 1, audio_data)
    
    # Aplicar filtro pasa bajas para eliminar ruido de alta frecuencia
    nyquist = 0.5 * sample_rate
    low_cutoff = 3000 / nyquist  # Frecuencia de corte en Hz (por ejemplo, 3000 Hz)
    b, a = signal.butter(5, low_cutoff, btype='low')
    audio_data = signal.filtfilt(b, a, audio_data)
    
    # Normalizar el audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Reducción de ruido con librosa
    audio_data = librosa.effects.preemphasis(audio_data, coef=preemphasis_coeff)
    
    # Calcula la frecuencia del audio grabado
    frequency = calculate_frequency(audio_data, sample_rate)
    print("Frecuencia del audio: {:.2f} Hz".format(frequency))
    
    return audio_data

def main():
    global mic_active
    while True:
        print("Presiona '1' para iniciar la grabación del micrófono.")
        key = input()
        if key == '1' and not mic_active:
            # Inicia la grabación en un hilo separado
            threading.Thread(target=record_audio).start()
        elif key == '2' and mic_active:
            mic_active = False
            while mic_active:
                time.sleep(0.1)  # Esperar a que el hilo de grabación se detenga
            print("¿Deseas hacer otra grabación? (s/n)")
            another_recording = input().strip().lower()
            if another_recording == 'n':
                break   
if __name__ == "__main__":
    main()