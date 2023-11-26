# IA2_TC3007C - Implementación de un modelo de deep learning (Portafolio de implementación)
# Karen Cebreros López - A01704254

## Correcciones

- Se agregó la documentación de la mejora del modelo en el apartado **Modelo de keras que utilicé** dentro del readme, donde se pueden leer las características y resultados de mi primer modelo en el apartado **Primer modelo** y las mejoras que le hice a mi modelo final a partir de éstos en el apartado **Modelo final (mejoras al primer modelo)**.
- Así mismo, dentro del archivo **create_model.py** agregué la función **createInitialModel()** en una condicional, que contiene el código que utilicé para crear mi primer modelo. (Ésta se encuentra comentada para no volver a crearlo, pero en caso de que se quiera correr, solo hay que descomentar la línea número 30 del código del main y poner un 0 en la variable **modelo_a_cargar** que se encuentra en la línea 24 del main).
- Por último, también agregué mi primer modelo junto con su historial en la carpeta **Model and history** con los nombres *MobileNet_model1_signs.h5* y *MobileNet_model1_history.json* respectivamente. Si se quieren ver los resultados del primer modelo, solo se tiene que poner un 0 en la variable **modelo_a_cargar** que se encuentra en la línea 24 del main.

## Descripción
*Antes que nada, quiero decir que para este entregable tenía un trabajo hecho sobre un modelo para clasificación de audio de muchos idiomas, para poder identificar cual es el que se habla. Sin embargo, por cuestiones de tiempo (era muy pesado hacer varios entrenamientos de esto en mi computadora), decidí mejor hacer mi entregable con mi idea original, aunque después planeo después agregar mi segundo proyecto a este mismo repo : )*

Ahora si...

Para este entregable decidí hacer un modelo de deep learning que pudiera predecir el alfabeto de lenguaje de señas a través de imágenes, utilizando transfer learning y mis adaptaciones del modelo para la clasificacióm.

![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/sign_alphabet.png)

## Especifiaciones para correr el código
En el main está todo listo para poner el código a prueba, cargar el modelo final con su historial (nombrados como *MobileNet_finalmodel_signs.h5* y *MobileNet_finalmodel_history.json*), que se encuentran dentro de la carpeta "Model and history" para hacer predicciones. Esto lo define una variable llamada **modelo_a_cargar** (que por default está en 1 para probar el modelo final) que se encuentra en la línea 24 del main.

Si se desea volver a crear el modelo final, se tendría que descomentar la línea número 30 del código del main, donde se encuentra la función **createFinalModel()** dentro de una condiconal. Si se desea volver a crear el modelo inicial, se tendría que descomentar esa misma línea (30) del código, donde se encuentra la función **createInitialModel()** dentro de la misma condicional. Y aparte se tendría que inicializar **modelo_a_cargar** en 0, ya que este indica que se quiere trabajar con el primer modelo.

Para la preparación de los datos, se utiliza la misma lógica, solamente que la línea que se tendría que descomentar es la línea de código 27 del main, donde hay una condicional para **prepareFiles()**, dependiendo del modelo que se quiera utilizar. (1 en *modelo_a_cargar* para el modelo final y 0 en *modelo_a_cargar* para el modelo inicial. (Personalmente no recomiendo esto porque puede tardar mucho).

**(NOTA: los nombres por default que tienen los archivos del modelo y su historial para modelo final, son: *MobileNet_finalmodel_signs.h5* y *MobileNet_finalmodel_history.json* respectivamente. De igual forma los nombres por default que tienen los archivos del modelo y su historial para el primer modelo, son: *MobileNet_model1_signs.h5* y *MobileNet_model1_history.json* respectivamente. En caso de que alguno de los modelos se vuelva a crear, se sobreescriben estos archivos con sus respectivos nombres).**

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
Vienen dos archivos que ya contienen lista la información y por separado para el conjunto de datos de entrenamiento y de prueba. Sin embargo, yo si hice una preparación de los datos, para cambiar su tamaño y su forma, para mejorar el desempeño de mi modelo.

## Modelo de keras que utilicé
Para el transfer learning decidí utilizar MobileNet. 

MobileNet es un modelo del framework 'keras', que utiliza una operación llamada convolución separable en profundidad y tiene una arquitectura muy liviana. Es por ello que trabaja muy bien y muy rápido con imágenes y entornos con recursos limitados. Al hacer mi script en Google Colab, decidí que este era el ideal para trabajar de forma eficiente y tener buenos resultados. Sin embargo, cabe aclarar que MobileNet debe recibir un 'input_shape' de mínimo 30x30x3; y al ser mis imágenes de un tamaño original de 28x28, tuve que hacer un proceso de 'resize and reshape', antes de hacer mi modelo.

***(NOTA: evidentemente, solamente utilicé parte de la estructura de la arquitectura de MobileNet y yo personalicé la última parte que vendría siendo el clasificador de 24 clases para el alfabeto en lenguaje de señas)***.

### Primer modelo
La parte que yo modifiqué del modelo, fue la parte final y el clasificador. Estas son las capas que agregué, las modificaciones y sus características:

Datos:
- Imágenes de 40x40x3 (para este primer modelo, solo las hice tantito más grandes de lo que ya venía).

Características:
- Epochs: 10
- Learning rate: 0.0003
- train steps_per_epoch: 100
- validation_steps: 80
- test steps: 40

Capas:
- Para capturar una cierta variedad de características de las imágenes y hacer mi modelo más preciso, agregué dos capas **Dense()** con 128 neuronas y una función de activación de tipo *relu*. (Evidentemente en la capa final, la función de activación cambia porque se trata de un ejercicio de multiclase).
- Posteriormente agregué una capa **Flatten()**, para "aplanar" y convertir los datos a 1 dimensión.
- Y finalmente, agregué una capa **Dense()** con 24 neuronas (porque tengo 24 clases) y una función de activación de tipo *softmax* porque como dije anteriormente, se trata de un ejercicio multiclase.

#### Resultados:
##### Entrenamiento:
Accuracy = 32.15% || Loss = 2.2727

##### Validación:
Accuracy = 26.44% || Loss = 2.4433

##### Prueba:
Accuracy = 24.53% || Loss = 2.5494

Como se puede observar, este modelo no fue para nada bueno y a pesar de que traté de jugar con los hiperparámetros y las capas, el 'accuracy' no aumentó mucho. Fue por esto que para mi siguiente modelo, decidí trabajar con imágenes más grandes y hacer otras cuantas modificaciones para que mejorara la precisión de mi modelo.

***(NOTA: nuevamente, el código de la función que crea este modelo, se llama "createInitialModel()" y se encuentra comentada en la línea 30 del código del main dentro de una condicional. En caso de que se quiera trabajar con este modelo, lo primero que hay que hacer es poner un 0 en la variable modelo_a_cargar en la línea 24 del main. Luego, en caso de querer crear de nuevo el primer modelo, se tendría que hacer es la línea 30 del main. Y en caso de que se quieran volver a preparar los datos de este modelo, se tendría que descomentar la línea número 27 del main).***

### Modelo final (mejoras al primer modelo)
Después de observar los malos resultados de mi primer modelo, decidí hacerle unas cuantas modificaciones para incrementar su precisión en cuanto a las predicciones. Estas fueron las mejoras que le hice:

Datos:
- Imágenes de 80x80x3 (para este modelo, las agrandé el doble de lo que ya las tenía en mi primer modelo).

Características:
- Epochs: 10
- Learning rate: 0.0001
- train steps_per_epoch: 100
- validation_steps: 40
- test steps: 20

Capas:
- Para la reducción de parámetros y por ende, para la prevención de sobre ajuste (ya que me llegó a pasar esto con mi primer modelo), primeramente agregué una capa de **GlobalAveragePooling2D()**.
- Luego, de nuevo para capturar una cierta variedad de características de las imágenes y hacer mi modelo más preciso, puse ahora una capa **Dense()**, pero esta vez con 256 neuronas y una función de activación de tipo *relu*. (Evidentemente en la capa final, la función de activación cambia porque se trata de un ejercicio de multiclase).
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

***(NOTA: ya no hice más grandes las imágenes debido a que es bastante pesado procesar tantas imágenes así y tardaba mucho).***


## Conclusión
Como conclusión, quiero decir que considero que utilizar transfer learning es una técnica bastante buena cuando se trabaja con ciertos sets de datos, que son grandes y requieren de una estructura bastante robusta para ser entrenados en un modelo.
Si bien considero que este proyecto que elegí fue bastante sencillo, al trabajar en éste me di cuenta de cual es la verdadera importancia de los recursos y los datos que se poseen para trabajar en esto. Pues si trabajando solo con imágenes tuve que adaptar bastante mis datos y mi modelo para que pudiera funcionar bien con los recursos que tenía a la mano, para desarrollar el proyecto que originalmente tenía en mente, evidentemente necesito buscar la forma de poder ya sea adaptar el modelo a ciertos recursos, o trabajar con otras herramientas para poder llevarlo a cabo como planeaba.
