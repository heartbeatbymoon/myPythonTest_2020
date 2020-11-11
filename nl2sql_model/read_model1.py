#encoding=utf-8
import tensorflow as tf
from keras.models import load_model
from keras.models import Model

model = load_model("G:\\models\\task1_best_model.h5")
model.summary()



