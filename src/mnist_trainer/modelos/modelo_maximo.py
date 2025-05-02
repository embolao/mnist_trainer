from tensorflow.keras import layers, models, regularizers


def create_max_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Crea un modelo CNN profundo y robusto para MNIST.
    """

    model = models.Sequential(name="Modelo_Maximo_MNIST")

    # Primer bloque convolucional
    model.add(
        layers.Conv2D(
            64, (3, 3), padding="same", activation="relu", input_shape=input_shape
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # Segundo bloque convolucional
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.35))

    # Tercer bloque convolucional
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.45))

    # Capa totalmente conectada
    model.add(layers.Flatten())
    model.add(
        layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Capa de salida
    model.add(layers.Dense(num_classes, activation="softmax"))

    # Compilación
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
