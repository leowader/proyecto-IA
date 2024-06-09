from cProfile import label  # Para mostrar los mensajes en la interfaz
import tkinter as tk  # Para la interfaz gráfica
from PIL import Image, ImageTk  # Para manejar imágenes
import sounddevice as sd
import numpy as np  # Para manejar los datos de audio
from scipy.io.wavfile import write  # Para manejar los datos de audio
from pydub import AudioSegment
import threading  # Para manejar la grabación de audio en un hilo separado.
import os  # Para manejar el guardado de archivo en los directorios
from scipy.fft import fft, fftfreq  # Para el análisis de frecuencias
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logic.libs.training import saveInputs
from logic.libs.training import saveInputs

# Variables globales para la grabación de audio
is_recording = False
recording_thread = None
filename_wav = "output.wav"
directory = "./audio/"  # Cambia esto al directorio deseado
def reduce_audio(audio_data_mono, fs, freq):
    # Encontrar el índice del primer pico de frecuencia alta
    peak_index = np.argmax(np.abs(audio_data_mono))

    # Calcular el índice correspondiente a un segundo después del pico
    samples_per_second = fs
    one_second_after_peak_index = int(peak_index + samples_per_second)

    # Cortar el audio desde el primer pico hasta un segundo después de ese pico
    reduced_audio = audio_data_mono[peak_index:one_second_after_peak_index]
    # Guardar el audio reducido
    write("reduced_output.wav", fs, reduced_audio.astype(np.int16))
    print("tamaño vector reducido",len(reduced_audio))
    print("Audio reducido guardado como 'reduced_output.wav'")
    saveInputs(inputs= np.array(reduced_audio).tolist())
def toggle_recording():
    global is_recording, recording_thread

    if is_recording:
        is_recording = False
        status_label.config(text="Audio grabado")
        # save_audio()
    else:
        is_recording = True
        status_label.config(text="Grabando audio")
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.start()

def record_audio():
    global is_recording, filename_wav
    fs = 22050  # Frecuencia de muestreo
    duration = 5  # Duración máxima (segundos)

    # Grabar audio
    print("Grabando...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    # Esperar hasta que se complete la grabación o se detenga manualmente
    for _ in range(int(duration * 10)):
        if not is_recording:
            break
        sd.sleep(100)

    sd.stop()  # Detener la grabación si se detiene manualmente
    write(filename_wav, fs, audio_data)
    print("Grabación completada")

        # Analizar la frecuencia más alta
    audio_data_mono = np.mean(audio_data, axis=1)  # Convertir a mono
    N = len(audio_data_mono)
    
    yf = fft(audio_data_mono)
    xf = fftfreq(N, 1 / fs)

    idx = np.argmax(np.abs(yf))
    freq = abs(xf[idx])
    print(f"Frecuencia más alta: {freq} Hz")
    print("tamaño vector",N)
  
    reduce_audio(audio_data_mono, fs, freq)

#-----------------------------------------------INTERFAZ--------------------------------------------------

# Crear la ventana principal
root = tk.Tk()
root.title("Reconocimiento de moneda")

# Ajustar el tamaño de la ventana (ancho x alto)
root.geometry("400x400")

# Cargar la imagen del micrófono
try:
    mic_image = Image.open("./img/microred.png")
    mic_image = mic_image.resize((100, 100), Image.Resampling.LANCZOS)
    mic_photo = ImageTk.PhotoImage(mic_image)
except FileNotFoundError:
    print("No se encontró la imagen del micrófono.")
    root.destroy()

# Crear un lienzo para el botón circular
canvas = tk.Canvas(root, width=100, height=100)
canvas.pack(expand=True)  # Permite que el lienzo se expanda

# Crear un botón con la imagen del micrófono y ajustar el tamaño del botón con padding
button = tk.Button(root, image=mic_photo, borderwidth=0, padx=20, pady=20, command=toggle_recording)
button_window = canvas.create_window(50, 50, window=button)

# Centrar el lienzo en la ventana principal
canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# Crear una etiqueta para mostrar el estado del audio
label = tk.Label(root, text="")
label.pack(pady=20)

# Crear una etiqueta para mostrar el estado del audio
status_label = tk.Label(root, text="")
status_label.pack(pady=20)

# Ejecutar el bucle principal de la aplicación
root.mainloop()