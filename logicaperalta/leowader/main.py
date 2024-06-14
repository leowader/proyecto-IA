import os
from tkinter import *
from tkinter import filedialog
import pyaudio
import wave
import librosa
import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk

RUTA_MODELO = "entrenamiento.h5"
MUESTRAS = 22050

class _Detectar_nota:
    modelo = None
    array_notas = ["100", "200", "500"]
    instancia = None
    
    def predecir(self, ruta_archivo):
        MFCCs = self.pre_procesado(ruta_archivo)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        predicciones = self.modelo.predict(MFCCs)
        indice_predecir = np.argmax(predicciones)
        palabra_predecir = self.array_notas[indice_predecir]
        return palabra_predecir

    def pre_procesado(self, ruta_archivo, num_mfcc=13, n_fft=2048, hop_length=512):
        senal, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)
        if len(senal) > MUESTRAS:
            senal = senal[:MUESTRAS]
        mfccs = librosa.feature.mfcc(y=senal, sr=frecuencia_muestreo, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        return mfccs.T

def cargar_audios(ruta_carpeta):
    return [os.path.join(ruta_carpeta, archivo) for archivo in os.listdir(ruta_carpeta) if archivo.endswith('.wav')]

def seleccionar_audio(ruta_audio):
    CHUNK = 1024
    try:
        wf = wave.open(ruta_audio, 'rb')
        audio = pyaudio.PyAudio()

        # Configurar el stream de salida
        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

        # Leer y reproducir el audio en chunks
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)

        # Detener y cerrar el stream de audio
        stream.stop_stream()
        stream.close()
        audio.terminate()

    except Exception as e:
        print(f"Error al reproducir el archivo WAV: {e}")

def verificarSonido(ruta_audio):
    detectarNota = detectar_nota()
    resultado = detectarNota.predecir(ruta_audio)
    print("La moneda es:", resultado)
    return resultado

def mostrar_resultado(resultado):
    lblResultado.config(text="La moneda es: " + resultado, bg='white')
    image_path = "./assets/" + resultado + ".jpg"
    img = Image.open(image_path)
    img = img.resize((100, 100), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def cargar_carpeta(ruta_carpeta):
    archivos = cargar_audios(ruta_carpeta)
    lista_audios.delete(0, END)  # Limpiar lista actual
    for archivo in archivos:
        lista_audios.insert(END, archivo)

def cargar_carpeta_100():
    cargar_carpeta('./Sonidos/100')

def cargar_carpeta_200():
    cargar_carpeta('./Sonidos/200')

def cargar_carpeta_500():
    cargar_carpeta('./Sonidos/500')

def seleccionar_audio_seleccionado():
    seleccionado = lista_audios.get(ACTIVE)
    seleccionar_audio(seleccionado)
    resultado = verificarSonido(seleccionado)
    mostrar_resultado(resultado)

def grabar_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    duracion = 3
    archivo = "audio.wav"

    # INICIAMOS "pyaudio"
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # INICIAMOS GRABACION
    print("Grabando...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duracion)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Grabación terminada")

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

    # Mostrar en la interfaz gráfica
    lblResultado.config(text="Audio grabado: " + archivo, bg='white')

    # Simular reconocimiento
    resultado = verificarSonido(archivo)
    mostrar_resultado(resultado)

def reproducir_audio_seleccionado():
    seleccionado = lista_audios.get(ACTIVE)
    if seleccionado:
        seleccionar_audio(seleccionado)
        resultado = verificarSonido(seleccionado)
        mostrar_resultado(resultado)
    else:
        print("No se ha seleccionado ningún archivo de audio.")

# Configuración de la interfaz gráfica
ventaMain = Tk()
ventaMain.geometry('800x500')
ventaMain.configure(bg='white')
ventaMain.title('RECONOCIMIENTO DE MONEDAS')

# Panel izquierdo con botones
panel_izquierdo = Frame(ventaMain, bg='white')
panel_izquierdo.pack(side=LEFT, padx=20, pady=20, fill=Y)

lblTitulo = Label(panel_izquierdo, text="Seleccionar audio y reconocer moneda", bg='white', font=('Helvetica', 16))
lblTitulo.pack(pady=10)

imagen_boton = Image.open("./assets/microred.png")
imagen_boton = imagen_boton.resize((100, 100), Image.LANCZOS)
boton_imagen = ImageTk.PhotoImage(imagen_boton)

btnGrabar = Button(panel_izquierdo, image=boton_imagen, command=grabar_audio, bd=0, relief=FLAT, bg='white')
btnGrabar.pack(pady=10)

btnReproducirSeleccionado = Button(panel_izquierdo, text="Reproducir y reconocer", command=reproducir_audio_seleccionado)
btnReproducirSeleccionado.pack(pady=10)

btnCargar100 = Button(panel_izquierdo, text="Cargar Audios 100", command=cargar_carpeta_100)
btnCargar100.pack(pady=10)

btnCargar200 = Button(panel_izquierdo, text="Cargar Audios 200", command=cargar_carpeta_200)
btnCargar200.pack(pady=10)

btnCargar500 = Button(panel_izquierdo, text="Cargar Audios 500", command=cargar_carpeta_500)
btnCargar500.pack(pady=10)

# Panel derecho con lista de audios y resultado
panel_derecho = Frame(ventaMain, bg='white')
panel_derecho.pack(side=LEFT, padx=20, pady=20, fill=BOTH, expand=True)

lista_audios = Listbox(panel_derecho, selectmode=SINGLE, width=100, height=20)
lista_audios.pack(pady=20, padx=20, side=TOP, fill=BOTH, expand=True)

lblResultado = Label(panel_derecho, text="", bg='white', font=('Helvetica', 16))
lblResultado.pack(pady=10)

panel = Label(panel_derecho, bg='white')
panel.pack(pady=20)

def detectar_nota():
    if _Detectar_nota.instancia is None:
        _Detectar_nota.instancia = _Detectar_nota()
        _Detectar_nota.modelo = tf.keras.models.load_model(RUTA_MODELO)
    return _Detectar_nota.instancia

def PaginaPrincipal():
    ventaMain.mainloop()

if __name__ == "__main__":
    PaginaPrincipal()
