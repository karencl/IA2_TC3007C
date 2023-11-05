# IA2_TC3007C - Implementación de un modelo de deep learning (Portafolio de implementación)

## Descripción
*Antes que nada, quiero decir que para este entregable tenía un trabajo hecho sobre un modelo para clasificación de audio de muchos idiomas, para poder identificar cual es el que se habla. Sin embargo, por cuestiones de tiempo (era muy pesado hacer varios entrenamientos de esto en mi computadora), decidí mejor hacer mi entregable con mi idea original, aunque después planeo después agregar mi segundo proyecto a este mismo repo : )*

Ahora si...

Para este entregable decidí hacer un modelo de deep learning que pudiera predecir el alfabeto de lenguaje de señas a través de imágenes, utilizando transfer learning y mis adaptaciones del modelo para la clasificacióm.

![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/sign_alphabet.png)

## Especifiaciones para correrlo
En el main está todo listo para poner el código a prueba, cargar el modelo con su historial que se encuentran dentro de la carpeta "Model and history" y hacer predicciones. Sin embargo, si se quiere crear un nuevi modelo, solamente se tiene que descomentar la función **createModel()** que se encuentra en la línea número 27 del código del main. Así mismo, si se desea cargar el nuevo modelo con su historial que se va a guardar, es necesario poner el nombre de estos en las variables **modelo_cargado** e **historial_cargado** respectivamente, que se encuentran en el main en las líneas 30 y 31 del código del main. 

*(NOTA: el nombre por default que tienen estos dos nuevos documentos, son: **MobileNet_signs_new.h5** y **MobileNet_model_history_new.json** respectivamente).*

Por último, en caso de que se quiera hacer el resize y el split de los datos desde cero, será necesario descomentar la línea 24 del código del main, donde se encuentra la función de **prepareFiles()**. Personalmente no recomiendo esto porque puede tardar mucho.

## Dataset
El dataset que utilicé para este entregable se llama "Sign Language MNIST", obtenido de: https://www.kaggle.com/datasets/datamunge/sign-language-mnist/data.
Lo que se busca con este data set es predecir el lenguaje de señas, a través de imágenes del alfabeto. (Cabe aclarar que en este caso, se cuentan con 24 clases (24 letras diferentes) en vez de 26, debido a que en el lenguaje de señas, para la *j* y la *z* se requieren movimientos especiales y evidentemente no es posible analizar esto con imágenes).

Especificaciones:
- 784 pixeles por muestr
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
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/history.png)

En la imagen de arriba podemos observar como fue el historial del entrenamiento y la validación. Al final, en el entrenamiento se obtuvo **accuracy = 98.34%** y **loss = 0.1594**. Y en la validación se obtuvo **accuracy = 90.32%** y **loss =0.3844**.

Pasamos a verlo en gráficas:
#### Plots: Train & Validation - Accuracy & Loss
- **Train accuracy VS Validation accuracy** 
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/train_val_acc.png)

- **Train loss VS Validation loss**
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/train_val_loss.png)

Al observar ambas gráficas anteriores, podemos ver que el modelo se está comportando de buena manera y que no hay overfitting en este.

### 2 - Evaluate
![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/test_acc.png)

En la imagen de arriba podemos observar como fue la evaluación de mi modelo con el conjunto de prueba. Podemos ver que se obtuvo **accuracy = 91.25%**, lo cual es bastante bueno y lo consideré aceptable para pasar a la siguiente etapa.

### 2 - Predict

Para las predicciones simplemente decidí hacer una figura que tuviera 10 imágenes aleatorias del datasete prueba y mostrar en el título en valor real de la clase a la que pertenecen (en este caso sería que letra del alfabeto en lenguaje de señas se está mostrando) y al lado el valor de la predicción para cada imágen; tal y como se muestra a continuación:

![alt text](https://github.com/karencl/IA2_TC3007C/blob/master/Images/predictions.png)

Evidentemente, por los resultados del entrenamiento, la validación y la evaluación de mi modelo, los resultados de las predicciones son bastante buenos. No obstante, si bien muchas de las predicciones son correctas, hay algunas que no lo son por el tamaño de las imágenes (como con la primera, que predijo que era una *g*, cuando en realidad era una *h* porque realmente son muy parecidas en el alfabeto de lenguaje de señas). Pues a pesar de que MobileNet, como dije anteriormente, es bueno para trabajar con imágenes, en la página dice que es recomendable usar tamaños de 160x160 para tener resultados realmente buenos, ya que trabajar con imágenes muy pequeñas puede no llegar a ofrecer los resultados esperados.

***(NOTA: como dije en un principio, yo no hice más grandes las imágenes debido a que trabajé desde Colab y la memoria RAM que Google ofrece, no es suficiente para procesar tantas imágenes tan grandes. La razón principal por la que trabajé en éste y no desde mi computadora, es por un problema que tuve al intentar descargar el modelo en macos).***


## Conclusión
Como conclusión, quiero decir que considero que utilizar transfer learning es una técnica bastante buena cuando se trabaja con ciertos sets de datos, que son grandes y requieren de una estructura bastante robusta para ser entrenados en un modelo.
Si bien considero que este proyecto es bastante sencillo, al trabajar con en éste me di cuenta de cual es la real importancia de los recursos que se poseen para trabajar en esto. Pues si trabajando solo con imágenes tuve que adaptar mi modelo para que pudiera funcionar con los recursos que tenía a la mano, para desarrollar el proyecto que originalmente tenía en mente, evidentemente necesito buscar la forma de poder ya sea adaptar el modelo a ciertos recursos, o trabajar con otras herramientas para poder llevarlo a cabo como planeaba.
