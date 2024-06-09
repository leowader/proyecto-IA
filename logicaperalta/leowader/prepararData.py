import librosa 
import os
import json
Dataset="Sonidos"
Json= "datasetudio.json"
muestras_utilizadas=22050
def _PrepararJson_(Dataset,Json,n_mfcc=13,hop_leng=512,n_ff=2048):
    datos={
        "monedas_usadas":[],"labels":[],"MFCC":[],
        "archivos":[]
    }
    
    for i,(ruta,nombre,nombre_arch) in enumerate (os.walk(Dataset)):
        if ruta is not Dataset:
            nota=ruta.split("/")[-1]
            datos["monedas_usadas"].append(nota)
            print(f"procesando: /{nota}")
            for x in nombre_arch:
                archivo_nota=os.path.join(ruta,x)
                señal,sr=librosa.load(archivo_nota)
                if (len(señal)>= muestras_utilizadas):
                    señal=señal[:muestras_utilizadas]
                    MFFCs=librosa.feature.mfcc(señal,n_mfcc=n_mfcc,hop_leng=hop_leng,n_ff=n_ff)
                    datos["labels"].append(i-1)
                    datos["MFCC"].append(MFFCs.T.tolist())
                    datos["archivos"].append(archivo_nota)
    with open(Json,"w") as fp:
        Json.dump(datos,fp,indent=4)
        
if __name__ =="__main__":
    _PrepararJson_(Dataset,Json)
    print("termino de generar archivo json")
            