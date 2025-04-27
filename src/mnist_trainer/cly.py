# mnist_trainer/cli.py
import click

from mnist_trainer.train import train_mnist_model


@click.command()
def main():
    """Entrenar el modelo MNIST."""
    train_mnist_model()


if __name__ == "__main__":
    main()
