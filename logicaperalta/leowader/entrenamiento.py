import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
RUTA_DATASET="datasedaudio.json"
GUARDAR_MODELO= "modelook.h5"
EPOCAS=1000
BATCH_SIZE=32
CARPETAS=5
TASA_APRENDIZAJE=0.0001
# def cargar_dataset(ruta_datos):