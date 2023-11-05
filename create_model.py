# Librerías
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import json


def reshapeData(imgs, lbls):
    """ Hace un reshape de las imágenes 
    Args:
        images (DataFrame) - DataFrame de las imágenes
    Return:
        reshaped_images (ndarray) - arreglo con las imágenes listas
    """
    # Reshape para tenerlas en el formato que se necesita (en este caso: (x, 80, 80, 3))
    reshaped_images = np.reshape(imgs, (-1, 80, 80))[...,np.newaxis]
    reshaped_images = np.repeat(reshaped_images, 3, axis=-1)
    
    return reshaped_images, lbls.squeeze()


def getGenerator(imgs, lbls):
    """ Obtiene el ImageGenerator() de un dataset 
    Args:
        imgs (ndarray) - arreglo con las imágenes
        lbls (ndarray) - arreglo con las etiquetas de las imágenes
    Return:
        gen (ImageGenerator) - generador del dataset
    """
    gen = ImageDataGenerator().flow(x = imgs, y = lbls)
    return gen


def createModel():
    """ 
    Crea, entrena, valida, evalua y guarda el modelo
    """
    
    # Leemos los archivos   
    imgs_train_df = pd.read_csv('Dataset/resized_train.csv')
    lbls_train_df = pd.read_csv('Dataset/train_labels.csv')
    imgs_val_df = pd.read_csv('Dataset/resized_validation.csv')
    lbls_val_df = pd.read_csv('Dataset/validation_labels.csv')
    imgs_test_df = pd.read_csv('Dataset/resized_test.csv')
    lbls_test_df = pd.read_csv('Dataset/test_labels.csv')
    
    # Les hacemos un reshape a las imágenes
    print('Haciedo el reshape de las imágenes...')
    train_images, train_labels = reshapeData(imgs_train_df.values, lbls_train_df.values)
    val_images, val_labels = reshapeData(imgs_val_df.values, lbls_val_df.values)
    test_images, test_labels = reshapeData(imgs_test_df.values, lbls_test_df.values)

    # Utilizo el alfabeto para las clases, sin las letras que no vienen incluidas en el dataset
    alphabet = list(string.ascii_lowercase)
    alphabet.remove('j')
    alphabet.remove('z')

    # Hago el generador de cada conjunto de datos
    train_generator = getGenerator(train_images, train_labels)
    val_generator = getGenerator(val_images, val_labels)
    test_generator = getGenerator(test_images, test_labels)
    
    # Descargo el modelo
    print('Creando el modelo...')
    conv_base= MobileNet(weights = 'imagenet',
                        include_top = False,
                        input_shape = (80, 80, 3))

    # Hago el clasificador
    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256,activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(24,activation='softmax'))

    # Indico que no se va a entrenar todo el modelo
    conv_base.trainable = False

    print(model.summary())

    model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizers.Adam(learning_rate=0.0001),
                    metrics=['acc'])

    # Entreno y valido el modelo
    epochs = 10
    print('Entrenando el modelo...')
    history = model.fit(train_generator,
                        steps_per_epoch = 100,
                        epochs = epochs,
                        validation_data= val_generator,
                        validation_steps = 40)
    
    # Evalúo el modelo
    print('Evaluando el modelo...')
    test_loss, test_acc = model.evaluate(test_generator, steps = 20)
    history.history['test_loss'] = test_loss
    history.history['test_acc'] = test_acc

    # Guardo el modelo
    with open('Model and history/MobileNet_model_history_new.json', 'w') as f:
        json.dump(history.history, f)
    model.save('Model and history/MobileNet_signs_new.h5')
    
    print('Se ha creado un nuevo modelo dento de Model and history')
    