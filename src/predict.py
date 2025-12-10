"""
TensorFlow POC - Classificação de Dígitos MNIST
Script de inferência/predição
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os


def load_model(model_path="models/mnist_model.keras"):
    """Carrega o modelo treinado"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            "Execute train.py primeiro para treinar o modelo."
        )

    print(f"Carregando modelo de: {model_path}")
    model = keras.models.load_model(model_path)
    return model


def preprocess_image(image_path):
    """Pré-processa uma imagem para predição"""
    img = Image.open(image_path).convert("L")  # Converter para grayscale
    img = img.resize((28, 28))  # Redimensionar para 28x28

    img_array = np.array(img).astype("float32") / 255.0

    # Inverter cores se necessário (MNIST tem fundo preto)
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array

    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array


def predict_single_image(model, image_path):
    """Faz predição para uma única imagem"""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence, prediction[0]


def predict_from_mnist_test():
    """Demonstra predição usando imagens do dataset de teste"""
    print("Carregando imagens de teste do MNIST...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    model = load_model()

    # Selecionar 9 imagens aleatórias
    indices = np.random.choice(len(x_test), 9, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for idx, ax in zip(indices, axes.flat):
        img = x_test[idx]
        prediction = model.predict(np.expand_dims(img, 0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        actual_class = y_test[idx]

        ax.imshow(img.squeeze(), cmap="gray")
        color = "green" if predicted_class == actual_class else "red"
        ax.set_title(
            f"Pred: {predicted_class} ({confidence:.1%})\nReal: {actual_class}",
            color=color,
            fontsize=10
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("models/demo_predictions.png")
    print("Demonstração salva em: models/demo_predictions.png")
    plt.show()


def interactive_demo():
    """Demonstração interativa com imagens do dataset"""
    print("\n" + "=" * 50)
    print("Demonstração Interativa")
    print("=" * 50)

    model = load_model()

    # Carregar dados de teste
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)

    print("\nTestando 10 imagens aleatórias:")
    print("-" * 40)

    correct = 0
    indices = np.random.choice(len(x_test), 10, replace=False)

    for i, idx in enumerate(indices):
        img = x_test[idx]
        prediction = model.predict(np.expand_dims(img, 0), verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        actual_class = y_test[idx]

        status = "✓" if predicted_class == actual_class else "✗"
        if predicted_class == actual_class:
            correct += 1

        print(f"{i+1}. Predição: {predicted_class} | Real: {actual_class} | "
              f"Confiança: {confidence:.1%} {status}")

    print("-" * 40)
    print(f"Acurácia: {correct}/10 ({correct/10:.0%})")


def main():
    parser = argparse.ArgumentParser(description="MNIST Prediction Script")
    parser.add_argument(
        "--image", "-i",
        type=str,
        help="Caminho para uma imagem de dígito para classificar"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Executar demonstração com imagens do MNIST"
    )
    parser.add_argument(
        "--interactive", "-t",
        action="store_true",
        help="Executar teste interativo"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("TensorFlow POC - Predição MNIST")
    print("=" * 50)
    print(f"TensorFlow versão: {tf.__version__}")
    print()

    if args.image:
        if not os.path.exists(args.image):
            print(f"Erro: Arquivo não encontrado: {args.image}")
            return

        model = load_model()
        predicted_class, confidence, probabilities = predict_single_image(model, args.image)

        print(f"\nResultado da Predição:")
        print(f"  Dígito predito: {predicted_class}")
        print(f"  Confiança: {confidence:.2%}")
        print(f"\nProbabilidades por classe:")
        for i, prob in enumerate(probabilities):
            bar = "█" * int(prob * 20)
            print(f"  {i}: {bar} {prob:.2%}")

    elif args.demo:
        predict_from_mnist_test()

    elif args.interactive:
        interactive_demo()

    else:
        # Por padrão, executar demonstração interativa
        interactive_demo()


if __name__ == "__main__":
    main()
