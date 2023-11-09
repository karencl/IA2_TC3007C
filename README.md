# IA2_TC3007C - Implementación de un modelo de deep learning (Portafolio de implementación)
# Karen Cebreros López - A01704254

## Descripción
*Antes que nada, quiero decir que para este entregable tenía un trabajo hecho sobre un modelo para clasificación de audio de muchos idiomas, para poder identificar cual es el que se habla. Sin embargo, por cuestiones de tiempo (era muy pesado hacer varios entrenamientos de esto en mi computadora), decidí mejor hacer mi entregable con mi idea original, aunque después planeo después agregar mi segundo proyecto a este mismo repo : )*

Ahora si...

Para este entregable decidí hacer un modelo de deep learning que pudiera predecir el alfabeto de lenguaje de señas a través de imágenes, utilizando transfer learning y mis adaptaciones del modelo para la clasificacióm.

![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/sign_alphabet.png)

## Especifiaciones para correr el código
En el main está todo listo para poner el código a prueba, cargar el modelo con su historial que se encuentran dentro de la carpeta "Model and history" y hacer predicciones. Sin embargo, si se quiere crear un nuevo modelo, solamente se tiene que descomentar la función **createModel()** que se encuentra en la línea número 28 del código del main. Así mismo, si se desea cargar el nuevo modelo con su historial que se va a guardar, es necesario poner el nombre de estos en las variables **modelo_cargado** e **historial_cargado** respectivamente, que se encuentran en el main en las líneas 31 y 32 del código del main. 

*(NOTA: el nombre por default que tienen estos dos nuevos documentos, son: **MobileNet_signs_new.h5** y **MobileNet_model_history_new.json** respectivamente).*

Por último, en caso de que se quiera hacer el resize y el split de los datos desde cero, será necesario descomentar la línea 24 del código del main, donde se encuentra la función de **prepareFiles()**. Personalmente no recomiendo esto porque puede tardar mucho.

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

## Modelo de keras que utilicé
Para el transfer learning decidí utilizar MobileNet. 

MobileNet es un modelo del framework 'keras', que utiliza una operación llamada convolución separable en profundidad y tiene una arquitectura muy liviana. Es por ello que trabaja muy bien y muy rápido con imágenes y entornos con recursos limitados. Al hacer mi script en Google Colab, decidí que este era el ideal para trabajar de forma eficiente y tener buenos resultados. Sin embargo, cabe aclarar que MobileNet debe recibir un 'input_shape' de mínimo 30x30x3; y al ser mis imágenes de un tamaño original de 28x28, tuve que hacer un proceso de 'resize and reshape', antes de hacer mi modelo.

***(NOTA: evidentemente, solamente utilicé parte de la estructura de la arquitectura de MobileNet y yo personalicé la última parte que vendría siendo el clasificador de 24 clases para el alfabeto en lenguaje de señas)***.

### Primer modelo
La parte que yo modifiqué del modelo, fue la parte final y el clasificador. Estas son las capas que agregué y las modificaciones:

- Para la reducción de parámetros y por ende, para la prevención de sobre ajuste, primeramente agregué una capa de **GlobalAveragePooling2D()**. (En este primer modelo no agregué una capa densa después de ésta, porque al ser una arquitectura hecha para datasets de imágenes, pensaba que no iba a ser necesario. Sin embargo no fue así y es por eso que en mi segundo modelo, como una de las mejoras, agregué una capa densa después de ésta).
- Posteriormente agregué una capa **Flatten()**, para "aplanar" y convertir los datos a 1 dimensión.
- Y finalmente, agregué una capa **Dense()** con 24 neuronas (porque tengo 24 clases) y una función de activación de tipo *softmax* porque como dije anteriormente, se trata de un ejercicio multiclase.

#### Resultados:
##### Entrenamiento:
Accuracy = 87.5% || Loss = 0.6833

##### Validación:
Accuracy = 79.37% || Loss = 0.8527

##### Prueba:
Accuracy = 77.5% || Loss = 0.8872

***(NOTA): el código de la función que crea este modelo, se llama "createInitialModel()" y se encuentra comentada en la línea 27 del código del main. En caso de querer volverlo a correr, lo único que se tendría que hacer es descomentarla. El archivo de este modelo y su histrorial se encuentran guardados en la carpeta "Model and history", como "MobileNet_model1_signs.h5" y "MobileNet_model1_history.json". Cabe mencionar que aunque se vuelva a correr esta función, el modelo que se cargará en el main para ver las predicciones, es el modelo con el nombre "MobileNet_bestmodel_signs.h5" con su historial correspondiente "MobileNet_bestmodel_history.json". En caso de que se quieran ver las predicciones y las gráficas del primer modelo, será necesario poner el nombre de estos en las variables **modelo_cargado** e **historial_cargado** respectivamente, que se encuentran en el main en las líneas 31 y 32 del código del main).***

### Modelo final (mejoras al primer modelo)
Después de observar los resultados de mi primer modelo, decidí hacerle unas cuantas modificaciones para incrementar su precisión en cuanto a las predicciones. Estas fueron las mejoras que le hice:

- Lo primero no lo modifiqué. Es decir, dejé la capa de **GlobalAveragePooling2D()** para la reducción de parámetros.
- Luego ahora si, para capturar una cierta variedad de características de las imágenes y hacer mi modelo más preciso, agregué una capa **Dense()** con 256 neuronas y una función de activación de tipo *relu*. (Evidentemente en la capa final, la función de activación cambia porque se trata de un ejercicio de multiclase).
- La siguiente capa (**Flatten()**), la dejé para "aplanar" y convertir los datos a 1 dimensión.
- Y finalmente, evidentemente dejé la otra capa **Dense()** con 24 neuronas (porque tengo 24 clases) y una función de activación de tipo *softmax* para mi clasificador miltuclase.

#### Desempeño por etapas:
##### 1 - Fit
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/history.png)

En la imagen de arriba podemos observar como fue el historial del entrenamiento y la validación. Al final, en el entrenamiento se obtuvo **accuracy = 98.34%** y **loss = 0.1593**. Y en la validación se obtuvo **accuracy = 90.31%** y **loss =0.3844**.

Pasamos a verlo en gráficas:
- **Train accuracy VS Validation accuracy** 
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/train_val_acc.png)

- **Train loss VS Validation loss**
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/train_val_loss.png)

Al observar ambas gráficas anteriores, podemos ver que el modelo se está comportando de buena manera y que no hay overfitting en este.

##### 2 - Evaluate
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/test_acc.png)

En la imagen de arriba podemos observar como fue la evaluación de mi modelo con el conjunto de prueba. Podemos ver que se obtuvo **accuracy = 92.18%**, lo cual es bastante bueno y lo consideré aceptable para pasar a la siguiente etapa.

##### 2 - Predict

Para las predicciones simplemente decidí hacer una figura que tuviera 10 imágenes aleatorias del datasete prueba y mostrar en el título en valor real de la clase a la que pertenecen (en este caso sería que letra del alfabeto en lenguaje de señas se está mostrando) y al lado el valor de la predicción para cada imágen; tal y como se muestra a continuación:

![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/predictions.png)

Evidentemente, por los resultados del entrenamiento, la validación y la evaluación de mi modelo, los resultados de las predicciones son bastante buenos y mejores a las del primer modelo que desarrollé. 

No obstante, si bien muchas de las predicciones son correctas, hay algunas que no lo son por el tamaño de las imágenes (como con la primera, que predijo que era una *g*, cuando en realidad era una *h* porque realmente son muy parecidas en el alfabeto de lenguaje de señas). Pues a pesar de que MobileNet, como dije anteriormente, es bueno para trabajar con imágenes, en la página dice que es recomendable usar tamaños de 160x160 para tener resultados realmente buenos, ya que trabajar con imágenes muy pequeñas puede no llegar a ofrecer los resultados esperados y sobre todo si se trata de algo más complejo. En este caso, no hubo en realidad problema por ello, debido a que el dataset y el objetivo del proyecto son bastantes sencillos de trabajar.

***(NOTA: como dije en un principio, yo no hice más grandes las imágenes debido a que trabajé desde Colab y la memoria RAM que Google ofrece, no es suficiente para procesar tantas imágenes tan grandes. La razón principal por la que trabajé en éste y no desde mi computadora, es por un problema que tuve al intentar descargar el modelo en macos).***


## Conclusión
Como conclusión, quiero decir que considero que utilizar transfer learning es una técnica bastante buena cuando se trabaja con ciertos sets de datos, que son grandes y requieren de una estructura bastante robusta para ser entrenados en un modelo.
Si bien considero que este proyecto que elegí fue bastante sencillo, al trabajar en éste me di cuenta de cual es la verdadera importancia de los recursos que se poseen para trabajar en esto. Pues si trabajando solo con imágenes tuve que adaptar mi modelo para que pudiera funcionar con los recursos que tenía a la mano, para desarrollar el proyecto que originalmente tenía en mente, evidentemente necesito buscar la forma de poder ya sea adaptar el modelo a ciertos recursos, o trabajar con otras herramientas para poder llevarlo a cabo como planeaba.
