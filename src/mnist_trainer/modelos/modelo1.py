from tensorflow.keras import layers, models


def create_improved_model():
    model = models.Sequential()

    # Primera capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling para reducir las dimensiones

    # Segunda capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Capa de normalización por lotes
    model.add(layers.BatchNormalization())

    # Capa convolucional adicional para mayor complejidad
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # Aplanar la salida para pasar a la capa densa
    model.add(layers.Flatten())

    # Capa densa con Dropout
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.3))  # Dropout más fuerte

    # Capa densa adicional
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.3))  # Dropout adicional

    # Capa de salida
    model.add(layers.Dense(10, activation="softmax"))

    # Compilación del modelo
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
