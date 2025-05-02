from tensorflow.keras import layers, models


def create_advanced_model():
    model = models.Sequential()

    # Entrada: imágenes 28x28 con 1 canal (blanco y negro)
    model.add(layers.Input(shape=(28, 28, 1)))

    # Primera capa convolucional: 32 filtros 3x3 con activación ReLU
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(
        layers.BatchNormalization()
    )  # Normaliza activaciones para acelerar entrenamiento

    # Segunda convolución: más capacidad de extracción de características
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())

    # Reducción de dimensionalidad y prevención de overfitting
    model.add(layers.MaxPooling2D((2, 2)))  # Reduce tamaño de 28x28 → 14x14
    model.add(layers.Dropout(0.25))  # Apaga 25% de neuronas al azar

    # Segunda etapa convolucional con 64 filtros (más capacidad)
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.BatchNormalization())

    # Otra reducción de dimensionalidad
    model.add(layers.MaxPooling2D((2, 2)))  # Reduce de 14x14 → 7x7
    model.add(layers.Dropout(0.25))

    # Aplanamiento de la salida para la red densa
    model.add(layers.Flatten())

    # Capa totalmente conectada con 256 neuronas
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))  # Mayor regularización (apaga el 50%)

    # Capa de salida: 10 clases (dígitos 0–9) con softmax para probabilidad
    model.add(layers.Dense(10, activation="softmax"))

    # Compilar el modelo con optimizador Adam y función de pérdida adecuada
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
