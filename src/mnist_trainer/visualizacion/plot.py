import matplotlib.pyplot as plt


# Definimos la función para graficar las métricas
def plot_metrics(history):
    # Graficar precisión
    plt.plot(history.history["accuracy"], label="Entrenamiento")
    plt.plot(history.history["val_accuracy"], label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Precisión")
    plt.legend()
    plt.show()

    # Graficar pérdida
    plt.plot(history.history["loss"], label="Entrenamiento")
    plt.plot(history.history["val_loss"], label="Validación")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()
    plt.show()
