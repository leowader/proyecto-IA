import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


RUTA_DATASET = "./datasetudio.json"
GUARDAR_MODELO = "modelook.h5"
EPOCAS = 1000
BATCH_SIZE = 32 # vaya dando una respuesta entre cada epoca
CARPETAS = 20
TASA_APRENDIZAJE = 0.0001

def cargar_dataset(ruta_datos):
    print(ruta_datos)
    with open(ruta_datos,'r') as file:
        data = json.load(file)
    x = np.array(data['MFCC'])
    y = np.array(data['labels'])
    print('Datos cargados')
    return x,y    

def prepararDataset(ruta_datos,test_size =0.1,tam_validacion=0.1):
    x,y = cargar_dataset(ruta_datos)
    
    assert 0 < test_size < 1, "test_size debe ser una proporción entre 0 y 1"
    assert 0 < tam_validacion < 1, "tam_validacion debe ser una proporción entre 0 y 1"
    
    X_entrenamiento,X_prueba,Y_entrenamiento,Y_prueba = train_test_split(x,y,test_size=test_size)
    X_entrenamiento,X_validacion,Y_entrenamiento,Y_validacion = train_test_split(X_entrenamiento,Y_entrenamiento,test_size=test_size)
    
    X_entrenamiento = X_entrenamiento[...,np.newaxis]
    X_prueba = X_prueba[...,np.newaxis]
    
    X_validacion = X_validacion[...,np.newaxis]
    
    return X_entrenamiento,Y_entrenamiento,X_validacion,Y_validacion,X_prueba,Y_prueba

def construir_modelo(input_shape, loss='sparse_categorical_crossentropy', rate=0.0001):
    modelo = tf.keras.models.Sequential()
    
    # 1ra capa de conv
    modelo.add(tf.keras.layers.Conv2D(50, (3, 3), activation='relu',
                                      input_shape=input_shape,
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    modelo.add(tf.keras.layers.Dropout(0.6))  # Aumentada la tasa de Dropout

    # 2da capa de conv
    modelo.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    modelo.add(tf.keras.layers.Dropout(0.6))  # Aumentada la tasa de Dropout

    # Aplanar y capas densas
    modelo.add(tf.keras.layers.Flatten())
    modelo.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    modelo.add(tf.keras.layers.Dropout(0.6))  # Aumentada la tasa de Dropout

    # Capa de salida
    modelo.add(tf.keras.layers.Dense(5, activation='softmax'))

    optimizador = tf.keras.optimizers.Adam(learning_rate=rate)

    # Compilar modelo
    modelo.compile(optimizer=optimizador,
                   loss=loss,
                   metrics=['accuracy'])

    # Imprimir resumen del modelo
    modelo.summary()

    return modelo

def entrenamiento(modelo, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.001, patience=patience)

    # Entrenar modelo
    modelo.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_validation, y_validation),
               callbacks=[earlystop_callback])
    
    return modelo


def main():
    # generar conjuntos de entrenamiento, validación y prueba
    X_entrenamiento, y_entrenamiento, X_validacion, y_validacion, X_prueba, y_prueba = prepararDataset(RUTA_DATASET)

    # crear red
    input_shape = (X_entrenamiento.shape[1], X_entrenamiento.shape[2], X_entrenamiento.shape[3])
    modelo = construir_modelo(input_shape, rate=TASA_APRENDIZAJE)

    # Entrenar red
    entrenamiento(modelo, EPOCAS, BATCH_SIZE, CARPETAS, X_entrenamiento, y_entrenamiento, X_validacion, y_validacion)

    # Guardar modelo
    modelo.save(GUARDAR_MODELO)
    
if __name__ == "__main__":
    main()   