"""
TensorFlow POC - Classificação de Dígitos MNIST
Script de treinamento do modelo
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_data():
    """Carrega e pré-processa o dataset MNIST"""
    print("Carregando dataset MNIST...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizar os pixels para valores entre 0 e 1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adicionar dimensão do canal (grayscale)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Shape treino: {x_train.shape}")
    print(f"Shape teste: {x_test.shape}")
    print(f"Classes: {np.unique(y_train)}")

    return (x_train, y_train), (x_test, y_test)


def create_model():
    """Cria o modelo CNN para classificação"""
    model = keras.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Segunda camada convolucional
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten e camadas densas
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_training_history(history, save_path="models/training_history.png"):
    """Plota e salva o histórico de treinamento"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico de acurácia
    axes[0].plot(history.history["accuracy"], label="Treino")
    axes[0].plot(history.history["val_accuracy"], label="Validação")
    axes[0].set_title("Acurácia do Modelo")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico de loss
    axes[1].plot(history.history["loss"], label="Treino")
    axes[1].plot(history.history["val_loss"], label="Validação")
    axes[1].set_title("Loss do Modelo")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de treinamento salvo em: {save_path}")


def plot_sample_predictions(model, x_test, y_test, save_path="models/sample_predictions.png"):
    """Plota algumas predições de exemplo"""
    predictions = model.predict(x_test[:16])
    predicted_classes = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i].squeeze(), cmap="gray")
        color = "green" if predicted_classes[i] == y_test[i] else "red"
        ax.set_title(f"Pred: {predicted_classes[i]} | Real: {y_test[i]}", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Predições de exemplo salvas em: {save_path}")


def main():
    """Função principal de treinamento"""
    print("=" * 50)
    print("TensorFlow POC - Classificação MNIST")
    print("=" * 50)
    print(f"TensorFlow versão: {tf.__version__}")
    print(f"GPU disponível: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()

    # Criar diretório de modelos se não existir
    os.makedirs("models", exist_ok=True)

    # Carregar dados
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Criar modelo
    print("\nCriando modelo CNN...")
    model = create_model()
    model.summary()

    # Treinar modelo
    print("\nIniciando treinamento...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        verbose=1
    )

    # Avaliar modelo
    print("\nAvaliando modelo no conjunto de teste...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    print(f"Loss no teste: {test_loss:.4f}")

    # Salvar modelo
    model_path = "models/mnist_model.keras"
    model.save(model_path)
    print(f"\nModelo salvo em: {model_path}")

    # Gerar visualizações
    plot_training_history(history)
    plot_sample_predictions(model, x_test, y_test)

    print("\n" + "=" * 50)
    print("Treinamento concluído com sucesso!")
    print("=" * 50)


if __name__ == "__main__":
    main()
