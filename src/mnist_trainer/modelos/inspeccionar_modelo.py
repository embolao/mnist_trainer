# import numpy as np
# import tensorflow as tf
from tensorflow import keras


def inspeccionar_modelo(ruta_modelo):
    print(f"\n📦 Cargando modelo desde: {ruta_modelo}")
    model = keras.models.load_model(ruta_modelo)

    print("\n📐 Resumen de arquitectura:")
    model.summary()

    print("\n🔍 Capas detalladas:")

    for i, layer in enumerate(model.layers):
        print(
            f"  [{i}] {layer.name} ({layer.__class__.__name__}) "
            f"-> Salida: {layer.output_shape}"
        )

    print("\n🔢 Parámetros totales:")
    total_params = model.count_params()
    print(f"  Total: {total_params:,}")

    print("\n📊 Pesos por capa (solo muestra shapes):")
    for _, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:
            print(f"  {layer.name}: {[w.shape for w in weights]}")
        else:
            print(f"  {layer.name}: (sin pesos entrenables)")


if __name__ == "__main__":
    ruta = "modelo_mnist.keras"
    inspeccionar_modelo(ruta)
