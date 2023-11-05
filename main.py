'''
    Módulo 2 - Deep Learning
    Momento de Retroalimentación: Implementación de un modelo de deep learning - (Portafolio Implementación)

    Karen Cebreros López - A01704254
    5/11/2023
'''

# Librerías
from prepare_data import prepareFiles
from create_model import createModel, reshapeData
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import string
import random
import json


if __name__ == '__main__':
    # En caso de que se quieran volver a preparar los datos, se descomentan estas líneas
    #prepareFiles()
    #print('Datos listos')
    
    createModel()
    print('Se ha creado un nuevo modelo!')
    
    # Cargo el modelo y su historial
    modelo_cargado = 'MobileNet_signs.h5'
    model = load_model(f'Model and history/{modelo_cargado}')
    with open('Model and history/MobileNet_model_history.json', 'r') as f:
        history = json.load(f)
    print(f'Se ha cargado el modelo: {modelo_cargado}')
    
    epochs = len(history['acc']) + 1
    
    # Gráfica de train accuracy VS validation accuracy
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs), history['acc'], 'm',label='train accuracy')
    plt.plot(range(1, epochs), history['val_acc'], 'c', label='validation accuracy')
    plt.title('train accuracy VS validation accuracy')
    plt.legend()
    
    # Gráfica de train loss VS validation loss
    plt.figure(figsize=(10,5))
    plt.plot(range(1, epochs), history['loss'], 'm', label ='training loss')
    plt.plot(range(1, epochs), history['val_loss'], 'c', label = 'validation loss')
    plt.title('train loss VS validation loss')
    plt.legend()
    
    print('Final train accuracy:', history['acc'][-1])
    print('Final validation accuracy:', history['val_acc'][-1])
    print('Final test accuracy:', history['test_acc'])
    
    # Preparo los datos para las predicciones
    imgs_test_df = pd.read_csv('Dataset/resized_test.csv')
    lbls_test_df = pd.read_csv('Dataset/test_labels.csv')
    test_images, test_labels = reshapeData(imgs_test_df.values, lbls_test_df.values)
    # Utilizo el alfabeto para las clases, sin las letras que no vienen incluidas en el dataset
    alphabet = list(string.ascii_lowercase)
    alphabet.remove('j')
    alphabet.remove('z')
    
    # Genero imagenes aleatorioas del dataset de prueba
    random_images_idx = random.choices(range(len(test_labels)), k=10)
    random_images = [test_images[i][np.newaxis,...] for i in random_images_idx]

    # Hago las predicciones
    predictions = [model.predict(img).squeeze() for img in random_images]

    # Muestro los resultados
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(random_images[i].squeeze(), cmap='gray')
        ax.axis('off')

        ax.set_title(f"Pred: {alphabet[predictions[i].argmax()]} -- Real: {alphabet[test_labels[random_images_idx[i]]]}")

    plt.tight_layout()
    plt.show()