# mnist_trainer/model.py
from tensorflow.keras import layers, models


def create_model():
    model = models.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(28, 28, 1)),
            layers.Flatten(input_shape=(28, 28, 1)),  # Aplanar la imagen de 28x28
            layers.Dense(128, activation="relu"),  # Capa densa con 128 unidades
            layers.Dropout(0.2),  # Dropout para evitar sobreajuste
            layers.Dense(64, activation="relu"),
            layers.Dense(
                10, activation="softmax"
            ),  # Capa de salida con 10 unidades (una por cada dígito)
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
