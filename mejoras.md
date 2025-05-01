# Mejoras en el proyecto `mnist_trainer`

Este archivo detalla las mejoras realizadas en el proyecto `mnist_trainer`, específicamente en el proceso de carga y preprocesamiento de los datos MNIST.

## 1. Uso de `tf.data.Dataset` para mejorar el rendimiento

En lugar de cargar manualmente las imágenes y convertirlas en arrays de NumPy, se implementó el uso de `tf.data.Dataset`, que permite un manejo más eficiente de los datos y la posibilidad de realizar operaciones en paralelo. Esto mejora el rendimiento general del código, especialmente cuando trabajamos con conjuntos de datos grandes.

### ¿Cómo se mejoró?

- Se utilizaron las funciones `shuffle`, `batch` y `prefetch` de `tf.data.Dataset` para optimizar el flujo de datos durante el entrenamiento y la evaluación del modelo.
- Se eliminó la necesidad de cargar y convertir imágenes manualmente, utilizando directamente las capacidades de `tensorflow_datasets` para manejar el dataset.

## 2. Normalización de imágenes

La normalización de las imágenes (para que estén en el rango de [0, 1]) se realiza dentro del pipeline de TensorFlow, lo que mejora la eficiencia al evitar operaciones adicionales sobre los datos cargados.

### ¿Cómo se mejoró?

- Se definió una función de normalización dentro del pipeline de `tf.data.Dataset`, lo que permite realizar la normalización mientras los datos se cargan, sin necesidad de convertir a arrays de NumPy.

## 3. Uso de `AUTOTUNE` para optimizar el número de hilos

La opción `AUTOTUNE` se implementó en las funciones de `map` y `prefetch` para que TensorFlow pueda optimizar el uso de los recursos de la máquina y acelerar el preprocesamiento de datos.

### ¿Cómo se mejoró?

- `AUTOTUNE` permite que TensorFlow determine automáticamente el número de hilos necesarios para realizar el preprocesamiento y la carga de datos de manera eficiente en función del hardware disponible.

## 4. Pipeline de datos optimizado

La carga de los datos ahora es más eficiente y flexible gracias al uso de `tf.data.Dataset`. Además, el código ahora se beneficia de una mayor modularidad, lo que facilita su extensión para incluir más transformaciones o técnicas de preprocesamiento.

### Beneficios

- **Rendimiento**: La implementación con `tf.data` permite el procesamiento paralelo de los datos, lo que mejora el rendimiento y hace que el proceso sea más rápido.
- **Escalabilidad**: Este enfoque puede manejar conjuntos de datos más grandes sin requerir la carga completa en memoria.
- **Simplicidad**: El código es más claro y fácil de mantener, utilizando las herramientas adecuadas de TensorFlow.

## 5. Posibilidad de incluir Aumento de Datos (Data Augmentation)

Aunque en esta versión no se ha implementado el aumento de datos, el pipeline actual es flexible y se puede extender fácilmente para incorporar transformaciones de imágenes como rotaciones, traslaciones o cambios de escala usando las herramientas proporcionadas por TensorFlow (`tf.image`).

## Código de ejemplo mejorado

```python
import tensorflow as tf
import tensorflow_datasets as tfds

def prepare_data(batch_size=32):
    print("Cargando MNIST...")

    # Cargar el dataset MNIST usando tensorflow_datasets
    mnist = tfds.load("mnist", as_supervised=True)

    # Obtener particiones de entrenamiento y prueba
    train_data, test_data = mnist["train"], mnist["test"]

    # Normalización de imágenes y creación de un pipeline de datos
    def normalize_img(image, label):
        # Normalizar imágenes entre [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Aplicamos el preprocesamiento de normalización a los datos
    train_data = train_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = test_data.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

    # Usamos el shuffle, batch y prefetch para optimizar el flujo de datos
    train_data = train_data.shuffle(60000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_data, test_data
