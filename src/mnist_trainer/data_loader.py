# mnist_trainer/data_loader.py

import numpy as np
import tensorflow_datasets as tfds


def prepare_data():
    print("Cargando MNIST...")
    mnist = tfds.load("mnist", as_supervised=True)

    # Obtener las particiones de entrenamiento y prueba
    train_data, test_data = mnist["train"], mnist["test"]

    # Convertir las particiones de datos a arrays numpy
    X_train, y_train = [], []
    for img, label in train_data:
        X_train.append(img.numpy())
        y_train.append(label.numpy())

    X_test, y_test = [], []
    for img, label in test_data:
        X_test.append(img.numpy())
        y_test.append(label.numpy())

    # Convertir a arrays numpy
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Normalizar los valores entre 0 y 1
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    return (X_train, y_train), (X_test, y_test)
