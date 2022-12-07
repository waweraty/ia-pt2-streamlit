# IA Concentración 2
## Primer Momento de Retroalimentación

En la Jupyter Notebook contenida en el presente repositorio, se incluye un primer acercamiento con algoritmos de machine learning para predecir tanto la altura como el caudal del cuerpo de agua con base en el set de datos que se nos proporcionó. Los puntos cubiertos son los siguientes:

1.- Explicación de las herramientas y tecnologías a usar para trabajar con los datos, y por qué son la mejor combinación.

2.- Generación el modelo de almacenamiento de los datos relacionados al reto.

3.- Extracción del conjunto de datos y limpieza. 

4.- Scripts de configuración utilizados, así como una muestra pequeña de los datos cargados.

5.- Separación de datos en conjuntos de prueba y de entrenamiento utilizando el esquema k-fold cross validation.

6.- De acuerdo con la cantidad de datos que se tiene determine si es necesario utilizar un enfoque orientado a Big Data o no.

Las herramientas y tecnologías se van a utilizar para trabajar con los datos en la primera etapa, donde vamos a conocer los datos y desarrollar un primer modelo es python y posteriormente, cuando se necesite, incluir una instancia  EC2 de AWS Deep Learning Notebook (Python 3.8, Tensor Flow 2.8, Pytorch 1.10) que cuesta $0.2 dlls por hora para potenciar el poder de colab, plataforma en la que estaremos trabajando colaborativamente.

Al no tener una cantidad masiva de datos, en este primer acercamiento, es posible usar las herramientas recién mencionadas, posteriormente, si se planea trabajar con una mayor cantidad de datos, sería conveniente manipularlos a través de stream processing, extrayendo la información con minería de datos y disminuyendo la cantidad de almacenamiento y poder necesario para procesar.

En la primera fase utilizaremos como forma de almacenamiento google drive, ya que solo contamos con un archivo CSV el cual estamos tratando en Colab.

Para la segunda fase, en la que se trabajará con imágenes, se espera que estas se encuentren en un servidor externo debido a su tamaño. Haremos uso de la librería OpenCV (o una similar) quizás con alguna implementación del algoritmo YOLO (You Only Look Once) para la extracción de características y detección de objetos. Ambas librerías e implementaciones se trabajarán de la misma forma, a través de Colab, pero ahora con la ayuda de la herramienta de cómputo en la nube AWS. 

Para el desarrollo del primer modelo no se necesita usar un enfoque orientado a Big data, ya que solo tenemos un CSV, pero cuando se deban procesar las imágenes y sacar sus atributos, se deberá considerar el segundo enfoque.

Model weights: https://drive.google.com/drive/folders/1eDGM7q7y_Jy2_bUuCQBfJTGhf1cp_P-H?usp=share_link

