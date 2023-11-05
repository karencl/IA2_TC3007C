# Librerías
from PIL import Image
import pandas as pd
import numpy as np


def resize_image_from_pixels(pixels, target_size=(80, 80)):
    """ Hcae un resize de una imagen que se pasa en pixeles
    Args:
        pixels (ndarray) - un arreglo que los 784 pixeles de las imágenes originales
        target_size (lst) - la dimensión a la que se quiere hacer el rezise (en este casi 224x224)
    
    Returns:
        reized_image (ndarray) - la imagen con su nuevo tamaño, ya normalizada
    """
    # Primero los 784 pixeles que se pasan, se convierten a una imagen de 28x28
    image = Image.fromarray(pixels.reshape(28, 28).astype(np.uint8))

    # Y ahora si se hace el resize
    resized_image = image.resize(target_size)

    # Nueva imagen ya normalizada
    return np.asarray(resized_image).ravel() / 255.0


def split_val_test(x, y, pct=0.5):
    """ Divide uno de los archivos (el de test/validation), en dos sets: test y validation
    Args:
        x (ndarray) - set de las imágenes que se quieren dividir
        y (lst) - set de las etiquetas que se quieren dividir
        pct (float) - valor que determina la proporción que le tocará a cada parte
    
    Returns:
        x_val (ndarray) - imágenes del set de validación
        y_val (ndarray) - etiquetas del set de validación
        x_test (ndarray) - imágenes del set de prueba
        y_test (ndarray) - etiquetas del set de prueba
    """

    x_val = x[:int(len(x)*pct)]
    x_test = x[int(len(x)*pct):]
    y_val = y[:int(len(y)*pct)]
    y_test = y[int(len(y)*pct):]

    return x_val, x_test, y_val, y_test


def saveFile(a, file_name):
    """ Guarda un ndarray en un documento csv
    Args:
        a (ndarray) - arreglo que contiene ya sea las imágenes o las etiquetas
        file_name (string) - nombre con el que se quiere guardar el archivo
    """
    df = pd.DataFrame(a)
    df.to_csv(file_name, index=False) 
    print(f'{file_name} created successfully!')
    

def prepareFiles():
    """ 
    Se hace toda la preparación de datos. Esto incluye el 'resize' y el split.
    """
    
    # Se leen los archivos .csv que contienen los pixeles de las imgánes y sus respectivas etiquetas 
    train_df = pd.read_csv('Dataset/sign_mnist_train.csv')
    test_df = pd.read_csv('Dataset/sign_mnist_test_val.csv')

    # Se obtienen los pixeles y las etiquetas de las imágenes
    train_image_data = train_df.iloc[:, 1:]
    test_image_data = test_df.iloc[:, 1:]
    train_labels = np.array(train_df['label'])
    test_labels = np.array(test_df['label'])

    # Se hace el resize de imágenes
    train_images_resized = train_image_data.apply(lambda row: resize_image_from_pixels(row.values), axis=1, result_type='expand')
    test_images_resized = test_image_data.apply(lambda row: resize_image_from_pixels(row.values), axis=1, result_type='expand')
    train_images = train_images_resized.values
    test_images = test_images_resized.values

    # Se hace la división para tener set de validación y prueba. (El de entrenamiento ya se tiene),
    val_images, test_images, val_labels, test_labels = split_val_test(test_images, test_labels, pct=0.6)

    # Se guardan los archivos
    saveFile(train_images, 'Dataset/resized_train.csv')
    saveFile(train_labels, 'Dataset/train_labels.csv')
    saveFile(test_images, 'Dataset/resized_test.csv')
    saveFile(test_labels, 'Dataset/test_labels.csv')
    saveFile(val_images, 'Dataset/resized_validation.csv')
    saveFile(val_labels, 'Dataset/validation_labels.csv')
    
    print('Datos listos')
