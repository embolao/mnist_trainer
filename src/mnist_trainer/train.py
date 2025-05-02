# mnist_trainer/train.py

# import matplotlib.pyplot as plt
# from tensorflow.keras import layers, models

from mnist_trainer.data.mejora_data_loader import prepare_data
from mnist_trainer.modelos.guardar_modelo import guardar_modelo
# from mnist_trainer.modelos.modelo1 import create_improved_model
# from mnist_trainer.modelos.modelo_avanzado import create_advanced_model
from mnist_trainer.modelos.modelo_maximo import create_max_model
from mnist_trainer.visualizacion.plot import plot_metrics


def train_model():
    # Preparar los datos
    train_data, test_data = prepare_data()

    # Crear el modelo
    model = create_max_model()

    # Entrenar el modelo
    try:
        history = model.fit(train_data, epochs=5, validation_data=test_data)
    except Exception as e:
        print("Error durante el entrenamiento:", e)

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(test_data)
    print(f"Precisión en test: {test_acc:.4f}")

    # Guardar el modelo entrenado
    guardar_modelo(model)
    print("Modelo guardado como modelo_mnist.keras")

    # Graficar precisión y pérdida
    plot_metrics(history)


if __name__ == "__main__":
    train_model()
