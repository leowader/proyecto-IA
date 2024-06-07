import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import numpy as np
from pymongo import MongoClient
client = MongoClient('mongodb+srv://leowader:251510...ld@cluster0.tjktldu.mongodb.net/moneda?retryWrites=true&w=majority&appName=Cluster0') 
db = client['moneda']
 
def createModel(numInputs):
    # Configuración de la red
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(numInputs,)),# numero de entradas
        tf.keras.layers.Dense(256,  activation='relu'), 
        tf.keras.layers.Dense(128, activation='relu'),  
        tf.keras.layers.Dense(1, activation='linear')  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mean_squared_error')
    return model

def normalize(inputs):
    normalized_inputs = []
    for sublist in inputs:
        maxNumber = max(sublist)
        normalized_sublist = [number / maxNumber for number in sublist]
        normalized_inputs.append(normalized_sublist)
    return normalized_inputs

def saveInputs(inputs): 
    collection = db['monedas']
    documento = {
    'pattern': inputs,
    }
    resultado = collection.insert_one(documento)
    print(f'Documento insertado con el id: {resultado.inserted_id}')
    
def training(inputs,outputs,numPatterns):
    datos = pd.read_excel("logic/data/500.xlsx")
    x = datos.filter(like='Hz') 
    y = datos.filter(like='Moneda')
    datosx = np.array(x, dtype=float)
    datosy = np.array(y, dtype=float)
    inputsNormalize = normalize(datosx)
    patrones=[[6286.05] * 132300]
    model=createModel(numPatterns)
    model.fit(np.array(normalize(inputs)), np.array(outputs), epochs=1000, verbose=False)
    # Datos de prueba
    X2 = np.array(normalize(patrones), dtype=float)
    # # x2Normalizado=[num / max(X2) for num in X2]

    predictions = model.predict(X2)
    print("prediciones",predictions)

   
    try:
        model.summary()
        model.save("logic/models/3segundos.keras")#guardar modelo
        print("Modelo guardado correctamente")
    except:
        print("No se guardo el modelo")

def deleteOne():
    result = collection.delete_one(filtro)

def simulation():
    model= tf.keras.models.load_model("logic/models/3segundos.keras")#importar modelo entrenado
    print("Simulacion")
    model.summary()
    x = pd.read_excel("logic/data/500.xlsx").filter(like="Hz")
    X2 = np.array([[6286.05] * 132300], dtype=float)#pasar por parametro esto
    # x2Normalizado=[num / max(np.array(x,dtype=float)) for num in X2]
    predictions=model.predict(X2)
    print("prediction",predictions)
    return predictions
# training([[6286.05] * 132300],[[1]],132300)