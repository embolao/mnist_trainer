import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model


def mostrar_activaciones(modelo, imagen):
    """
    Visualiza las activaciones de las capas convolucionales del modelo
    para una imagen dada.

    Args:
        modelo (tf.keras.Model): Modelo cargado
        imagen (np.array): Imagen de entrada (28x28)
    """
    try:
        # Preprocesamiento de la imagen
        if imagen.max() > 1.0:
            imagen = imagen / 255.0  # Normalizar a [0,1]

        # Añadir dimensiones de lote y canal
        imagen = np.expand_dims(imagen, axis=0)  # Añade dimensión de lote
        imagen = np.expand_dims(
            imagen, axis=-1
        )  # Añade dimensión de canal (escala de grises)

        # Obtener capas visualizables (convolucionales y de activación)
        capas_visualizables = []
        nombres_capas = []
        for capa in modelo.layers:
            if isinstance(
                capa,
                (
                    tf.keras.layers.Conv2D,
                    tf.keras.layers.MaxPooling2D,
                    tf.keras.layers.ReLU,
                    tf.keras.layers.Activation,
                ),
            ):
                capas_visualizables.append(capa.output)
                nombres_capas.append(capa.name)

        if not capas_visualizables:
            print("¡El modelo no tiene capas visualizables!")
            return

        # Crear modelo para extraer activaciones
        try:
            modelo_activaciones = tf.keras.models.Model(
                inputs=modelo.inputs, outputs=capas_visualizables
            )
        except Exception:
            # Método alternativo si falla el acceso estándar
            input_shape = modelo.layers[0].input_shape
            entrada = tf.keras.Input(shape=input_shape[1:])
            x = entrada
            for capa in modelo.layers:
                x = capa(x)
            modelo_activaciones = tf.keras.models.Model(
                inputs=entrada, outputs=capas_visualizables
            )

        # Obtener activaciones
        activaciones = modelo_activaciones.predict(imagen)

        # Visualizar cada capa
        for _, (act, nombre) in enumerate(zip(activaciones, nombres_capas)):
            print(f"\nCapa: {nombre} - Forma: {act.shape}")

            # Solo visualizar si tiene dimensiones espaciales
            if len(act.shape) == 4:
                n_filtros = act.shape[-1]
                filas = int(np.ceil(n_filtros / 8))
                plt.figure(figsize=(15, 2 * filas))
                plt.suptitle(f"Activaciones - {nombre}", y=1.05)

                for j in range(n_filtros):
                    plt.subplot(filas, 8, j + 1)
                    plt.imshow(act[0, :, :, j], cmap="viridis")
                    plt.axis("off")
                    plt.title(f"Filtro {j+1}")

                plt.tight_layout()
                plt.show()
            else:
                print(f"La capa {nombre} no produce salidas visualizables")

    except Exception as e:
        print(f"Error al visualizar activaciones: {str(e)}")


if __name__ == "__main__":
    try:
        # Cargar modelo
        modelo_path = "modelo_mnist.keras"
        print(f"Cargando modelo desde {modelo_path}...")
        modelo = load_model(modelo_path)
        print("¡Modelo cargado exitosamente!")
        modelo.summary()

        # Cargar datos MNIST
        (_, _), (X_test, _) = mnist.load_data()
        imagen_ejemplo = X_test[0]  # Primera imagen del conjunto de prueba

        # Mostrar imagen de ejemplo
        plt.imshow(imagen_ejemplo, cmap="gray")
        plt.title("Imagen de Entrada (MNIST)")
        plt.axis("off")
        plt.show()

        # Visualizar activaciones
        print("\nGenerando visualizaciones de activaciones...")
        mostrar_activaciones(modelo, imagen_ejemplo)

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo del modelo en {modelo_path}")
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
