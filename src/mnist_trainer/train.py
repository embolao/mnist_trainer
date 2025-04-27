# mnist_trainer/train.py

from mnist_trainer.data_loader import prepare_data
from mnist_trainer.model import create_model


def train_model():
    # Cargar los datos
    (X_train, y_train), (X_test, y_test) = prepare_data()

    # Crear el modelo
    model = create_model()

    # Entrenar el modelo
    model.fit(
        X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test)
    )

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    train_model()
