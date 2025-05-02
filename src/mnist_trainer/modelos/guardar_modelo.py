import os


def guardar_modelo(modelo, nombre_archivo="modelo_mnist.keras", carpeta="modelos"):
    os.makedirs(carpeta, exist_ok=True)  # Crea la carpeta si no existe
    ruta_completa = os.path.join(carpeta, nombre_archivo)
    modelo.save(ruta_completa)
    print(f"✅ Modelo guardado como: {ruta_completa}")
