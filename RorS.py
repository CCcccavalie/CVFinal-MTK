import cv2
import numpy as np
import keras
from keras.models import load_model
from numpy import argmax
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

def chooseClass(iml, imr):
    is_it_real = False
    iml = cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY)
    iml = cv2.resize(iml, (384, 512), interpolation=cv2.INTER_CUBIC)
    iml = np.expand_dims(iml, axis=-1)
    iml = np.expand_dims(iml, axis=0)/255
    model = load_model('./checkpoint/CNN_model_e07')
    estimationl = model.predict(iml, batch_size=1, steps=None)
    if estimationl[0][0]/estimationl[0][1] >25:
        is_it_real = True
    return is_it_real