import librosa as libro
import tensorflow as tf
import numpy as np
from tkinter import *
import pyaudio
import wave
import librosa
RUTA_MODELO = "modelook.h5"
MUESTRAS = 22050  # Considera audios superiores a 1 segs

class _Detectar_nota:

    modelo = None  # inicializamos el modelo como tipo none
    array_notas = [
        "1000",
        "200",
        "500",
    ]
    instancia = None
    
    def predecir(self, ruta_archivo):
        # ruta del archivo de audio para predecir
        # palabra clave predicha por el modelo

        # extraer MFCC
        MFCCs = self.pre_procesado(ruta_archivo)

        # necesitamos una matriz de 4 dim para alimentar el modelo para la predicción:
        # (muestras, pasos de tiempo, coeficientes, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # obtener la prediccion prevista
        predicciones = self.modelo.predict(MFCCs)
        indice_predecir = np.argmax(predicciones)
        palabra_predecir = self.array_notas[indice_predecir]
        return palabra_predecir


    def pre_procesado(self, ruta_archivo, num_mfcc=13, n_fft=2048, hop_length=512):
        """
        Extrae los MFCCs del archivo de audio especificado.

        Parámetros:
            ruta_archivo (str): Ruta del archivo de audio.
            num_mfcc (int): Número de coeficientes MFCC a extraer.
            n_fft (int): Tamaño de la ventana de análisis.
            hop_length (int): Tamaño del salto entre ventanas.

        Retorno:
            numpy.ndarray: Matriz de datos MFCC.
        """
        # Extraer los MFCCs del archivo de audio
        senal, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)

        # Asegurar la consistencia de la longitud de la señal
        if len(senal) > MUESTRAS:
            senal = senal[:MUESTRAS]

        # Extraer MFCCs
        mfccs = librosa.feature.mfcc(y=senal, sr=frecuencia_muestreo, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

        return mfccs.T

def detectar_nota():
    # verificar que se crea una instancia solo la primera vez que se llama a la función
    if _Detectar_nota.instancia is None:
        _Detectar_nota.instancia = _Detectar_nota()
        _Detectar_nota.modelo = tf.keras.models.load_model(RUTA_MODELO)

    return _Detectar_nota.instancia
#Ventana Principal
ventaMain = Tk()

ventaMain.geometry('300x200')
ventaMain.configure(bg='beige')

ventaMain.title('RECONOCIMIENTO DE MONEDAS')
lblTitulo = Label(ventaMain, text="Grabar caida moneda")

lblTitulo.place(relx=0.3, rely=0.1)
boton = Button(ventaMain, text="Grabar Audio", command=lambda: verificarSonido(), width=15, height=2)

boton.place(relx=0.3, rely=0.3)
boton.config(bg="#2354E2")

def verificarSonido():
    FORMAT=pyaudio.paInt16
    CHANNELS=1
    RATE=44100
    CHUNK=1024
    duracion=3
    archivo="audio.wav"

    # INICIAMOS "pyaudio"
    audio=pyaudio.PyAudio()
    stream=audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # INICIAMOS GRABACION
    print("grabando....")
    frames=[]
    for i in range(0, int(RATE / CHUNK * duracion)):
        data=stream.read(CHUNK)
        frames.append(data)
    print("grabacion terminada")

    # DETENEMOS GRABACION
    stream.stop_stream()
    stream.close()
    audio.terminate()
    # CREAMOS Y GUARDAMOS EL ARCHIVO DE AUDIO
    waveFile = wave.open(archivo, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    # 
    detectarNota = detectar_nota()
    detectarNota1 = detectar_nota()

    # comprobar que palabras clave apuntan al mismo objeto
    assert detectarNota is detectarNota1

    # Hacer una predicción
    resultado = detectarNota.predecir("./Sonidos/200/Grabación (50).wav")
    print(resultado)

    lblResultado = Label(ventaMain, text="La moneda  es: " + resultado)
    lblResultado.place(relx=0.25, rely=0.7)

def PaginaPrincipal():
    ventaMain.mainloop()

if __name__ == "__main__":
    PaginaPrincipal()
