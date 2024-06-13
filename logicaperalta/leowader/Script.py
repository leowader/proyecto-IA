import os
from pydub import AudioSegment

def convert_m4a_to_wav(folder_path):
    # Asegúrate de que la carpeta existe
    if not os.path.isdir(folder_path):
        print(f"La carpeta {folder_path} no existe.")
        return

    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith(".m4a"):
            # Crear ruta completa del archivo
            m4a_path = os.path.join(folder_path, filename)
            wav_filename = os.path.splitext(filename)[0] + ".wav"
            wav_path = os.path.join(folder_path, wav_filename)
            
            # Convertir de m4a a wav
            try:
                audio = AudioSegment.from_file(m4a_path, format="m4a")
                audio.export(wav_path, format="wav")
                print(f"Convertido: {m4a_path} -> {wav_path}")
            except Exception as e:
                print(f"Error al convertir {m4a_path}: {e}")

if __name__ == "__main__":
    folder_path = input("Introduce la ruta de la carpeta con los archivos .m4a: ").strip()
    print(f"Ruta proporcionada: {folder_path}")
    convert_m4a_to_wav(folder_path)