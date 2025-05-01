# mnist_trainer/cli.py
import click

from mnist_trainer.train import train_mnist_model


@click.command()
@click.option("--epochs", default=5, help="Número de épocas de entrenamiento.")
@click.option("--batch-size", default=32, help="Tamaño del batch.")
def main(epochs, batch_size):
    """Entrenar el modelo MNIST."""
    # Entrenar el modelo MNIST con opciones configurables.
    # Entrenar el modelo MNIST con opciones configurables.
    train_mnist_model(epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()
