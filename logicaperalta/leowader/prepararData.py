import librosa
import os
import json

Dataset = "Sonidos"
Json = "datasetudio.json"
muestras_utilizadas = 22050

def _PrepararJson_(Dataset, Json, n_mfcc=13, hop_length=512, n_fft=2048):
    print('Ruta del dataset:', os.path.abspath(Dataset))
    datos = {
        "monedas_usadas": [], "labels": [], "MFCC": [],
        "archivos": []
    }
    
    # Verifica si Dataset es una carpeta válida
    if not os.path.isdir(Dataset):
        print(f"El directorio {Dataset} no existe o no es un directorio.")
        return
    
    # Iterar sobre los archivos y carpetas en Dataset
    for i, (ruta, nombre, nombre_arch) in enumerate(os.walk(Dataset)):
        print('aqui llego:', ruta, nombre, nombre_arch)
        if ruta != os.path.abspath(Dataset):  # Compara con la ruta absoluta para mayor precisión
            nota = ruta.split(os.path.sep)[-1]
            datos["monedas_usadas"].append(nota)
            print(f"procesando: /{nota}")
            for x in nombre_arch:
                archivo_nota = os.path.join(ruta, x)
                try:
                    señal, sr = librosa.load(archivo_nota)
                except Exception as e:
                    print(f"Error al cargar el archivo {archivo_nota}: {e}")
                    continue
                if len(señal) >= muestras_utilizadas:
                    señal = señal[:muestras_utilizadas]
                    MFFCs = librosa.feature.mfcc(y=señal, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    datos["labels"].append(i - 1)
                    datos["MFCC"].append(MFFCs.T.tolist())
                    datos["archivos"].append(archivo_nota)
    
    with open(Json, "w") as fp:
        json.dump(datos, fp, indent=4)

if __name__ == "__main__":
    _PrepararJson_(Dataset, Json)
    print("termino de generar archivo json")