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
        tf.keras.Input(shape=(22050,)),# numero de entradas
        tf.keras.layers.Dense(32,  activation='relu'), 
        tf.keras.layers.Dense(164, activation='relu'),  
        tf.keras.layers.Dense(132, activation='sigmoid'),  
        tf.keras.layers.Dense(1, activation='relu')  
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error', metrics=['accuracy'])
    return model
def createInputs():
    coleccion_outputs = db['outputs']
    documents = [
    {'outputs': 0}, #   0   =   200 pesos
    {'outputs': 0},
    {'outputs': 0},
    {'outputs': 0},
    {'outputs': 0},
    {'outputs': 1},
    {'outputs': 1},
    {'outputs': 1},
    {'outputs': 1},
    {'outputs': 1},#    1   =   1000 pesos
    {'outputs': 1}]
    coleccion_outputs.insert_many(documents)
    
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
    'pattern': inputs
    }
    resultado = collection.insert_one(documento)
    print(f'Documento insertado con el id: {resultado.inserted_id}')
    
def getInputs():
    coleccion = db['monedas']
    outputs=db['outputs']
    
    # Obtener solo los valores de _id de cada documento
    ids = [document['pattern'] for document in coleccion.find()]
    out=[doc['outputs'] for doc in outputs.find()]
    return list(ids),list(out)


def training(inputs,outputs,numPatterns):
    inputs = normalize(np.array(inputs))
    outputs = np.array(outputs).reshape(-1, 1)
    model=createModel(numPatterns)
    model.fit(np.array(inputs), np.array(outputs), epochs=100,validation_split=0.2, verbose=True)
    try:
        model.summary()
        model.save("prueba2.keras")#guardar modelo
        print("Modelo guardado correctamente")
    except Exception as e:
        print("Error al guardar el modelo:", e)

def simulation(output):
    print("tamañooo",len(output))
    model= tf.keras.models.load_model("prueba2.keras")#importar modelo entrenado
    print("Simulacion")
    model.summary()
    # x = pd.read_excel("logic/data/500.xlsx").filter(like="Hz")
    X2 = np.array([output],dtype=float)#pasar por parametro esto
    # x2Normalizado=[num / max(np.array(x,dtype=float)) for num in X2]
    predictions=model.predict(X2)
    
    print("prediction",predictions)
    return predictions
# training([[6286.05] * 132300],[[1]],132300)
createInputs()