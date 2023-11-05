# IA2_TC3007C - Implementación de un modelo de deep learning (Portafolio de implementación)

## Descripción
*Antes que nada, quiero decir que para este entregable tenía un trabajo hecho sobre un modelo para clasificación de audio de instrumentos musicales. Sin embargo, por cuestiones de tiempo (era muy pesado hacer varios entrenamientos de esto en mi computadora), decidí mejor hacer mi entregable con mi idea original, aunque planeo después agregar mi segundo proyecto a este mismo repo : )*

Ahora si...

Para este entregable decidí hacer un modelo de deep learning que pudiera predecir el alfabeto de lenguaje de señas a través de imágenes, utilizando transfer learning y mis adaptaciones del modelo para la clasificacióm.

## Dataset
El dataset que utilicé para este entregable se llama "Sign Language MNIST", obtenido de: https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data.
Lo que se busca con este data set es predecir el lenguaje de señas, a través de imágenes del alfabeto. (Cabe aclarar que en este caso, se cuentan con 24 clases (24 letras diferentes) en vez de 26, debido a que en el lenguaje de señas, para la *j* y la *z* se requieren movimientos especiales y evidentemente no es posible analizar esto con imágenes).

Especificaciones:
- 784 pixeles por muestra
- Imágenes
- No. de clases: 24
- Tipo de datos: Enteros

Tamaño de cada conjunto:
- Set de entrenamiento: 27,455 imágenes
- Set de validción: 4,303 imágenes
- Set de prueba: 2,869 imágenes

## Especificaciones de uso del dataset para el entrenaimiento
Vienen dos archivos que ya contienen lista la información y por separado para el conjunto de datos de entrenamiento y de prueba.

## Modelo que utilicé
Para el transfer learning decidí utilizar MobileNet. 

MobileNet es un modelo del framework 'keras', que utiliza una operación llamada convolución separable en profundidad y tiene una arquitectura muy liviana. Es por ello que trabaja muy bien y muy rápido con imágenes y entornos con recursos limitados. Al hacer mi script en Google Colab, decidí que este era el ideal para trabajar de forma eficiente y tener buenos resultados. Sin embargo, cabe aclarar que MobileNet debe recibir un 'input_shape' de mínimo 30x30x3; y al ser mis imágenes de un tamaño original de 28x28, tuve que hacer un proceso de 'resize and reshape', antes de hacer mi modelo.

***(NOTA: evidentemente, solamente utilicé parte de la estructura de la arquitectura de MobileNet y yo personalicé la última parte que vendría siendo el clasificador de 24 clases para el alfabeto en lenguaje de señas)***.

## Desempeño por etapas
### 1 - Fit

En la imagen de arriba podemos observar como fue el historial del entrenamiento y la validación. Al final, en el entrenamiento se obtuvo **accuracy: ** y **loss: **. Y en la validación se obtuvo **accuracy: ** y **loss: **.

Pasamos a verlo en gráficas:
#### Plots: Train & Validation - Accuracy & Loss
- **Train accuracy VS Validation accuracy** 
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_3.png)

- **Train loss VS Validation loss**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_4.png)

Al observar ambas gráficas anteriores, podemos ver que el modelo se está comportando de buena manera y que no hay overfitting en este.

### 2 - Evaluate

En la imagen de arriba podemos observar como fue la evaluación de mi modelo con el conjunto de prueba. Podemos ver que se obtuvo **accuracy: **, lo cual es bastante bueno y lo consideré aceptable para pasar a la siguiente etapa.

### 2 - Predict

Para las predicciones simplemente decidí hacer una figura que tuviera 10 imágenes aleatorias del datasete prueba y mostrar en el título en valor real de la clase a la que pertenecen (en este caso sería que letra del alfabeto en lenguaje de señas se está mostrando) y al lado el valor de la predicción para cada imágen; tal y como se muestra a continuación:


Evidentemente, por los resultados del entrenamiento, la validación y la evaluación de mi modelo, los resultados de las predicciones son bastante buenos. No obstante, si bien muchas de las predicciones son correctas, hay algunas que no lo son por el tamaño de las imágenes; pues a pesar de que MobileNet, como dije anteriormente, es bueno para trabajar con imágenes, en la página dice que es recomendable usar tamaños de 160x160 para tener resultados realmente buenos, ya que trabajar con imágenes muy pequeñas puede no llegar a ofrecer los resultados esperados.

***(NOTA: como dije en un principio, yo no hice más grandes las imágenes debido a que trabajé desde Colab y la memoria RAM que Google ofrece, no es suficiente para procesar tantas imágenes tan grandes. La razón principal por la que trabajé en éste y no desde mi computadora, es por un problema que tuve al intentar descargar el modelo en macos).***


## Conclusión

