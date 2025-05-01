# mnist_trainer/train.py

from mnist_trainer.data.mejora_data_loader import prepare_data
from mnist_trainer.modelos.modelo1 import create_improved_model


def train_model():
    # Preparar los datos
    train_data, test_data = prepare_data()

    # Crear el modelo
    model = create_improved_model()

    # Entrenar el modelo
    try:
        model.fit(train_data, epochs=5, validation_data=test_data)
    except Exception as e:
        print("Error durante el entrenamiento:", e)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Precisión en test: {test_acc:.4f}")


if __name__ == "__main__":
    train_model()
