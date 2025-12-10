"""
TensorFlow POC - Interface Web
Aplicação Flask com interface visual para classificação de dígitos
"""

import os
import io
import base64
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# Variável global para o modelo
model = None


def load_model():
    """Carrega o modelo treinado"""
    global model
    model_path = "models/mnist_model.keras"

    if os.path.exists(model_path):
        print("Carregando modelo existente...")
        model = keras.models.load_model(model_path)
        return True
    return False


def train_model():
    """Treina um novo modelo se não existir"""
    global model
    print("Treinando novo modelo...")

    # Carregar dados
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)

    # Criar modelo
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

    # Salvar modelo
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_model.keras")
    print("Modelo treinado e salvo!")


def preprocess_canvas_image(image_data):
    """Processa imagem do canvas para predição"""
    # Decodificar base64
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    # Abrir imagem
    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # Redimensionar para 28x28
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Converter para array e normalizar
    img_array = np.array(img).astype("float32") / 255.0

    # Inverter (canvas tem fundo branco, MNIST tem fundo preto)
    img_array = 1 - img_array

    # Adicionar dimensões batch e canal
    img_array = np.expand_dims(img_array, axis=(0, -1))

    return img_array


@app.route("/")
def index():
    """Página principal"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint para predição"""
    global model

    if model is None:
        return jsonify({"error": "Modelo não carregado"}), 500

    try:
        data = request.json
        image_data = data.get("image")

        if not image_data:
            return jsonify({"error": "Nenhuma imagem enviada"}), 400

        # Processar imagem
        img_array = preprocess_canvas_image(image_data)

        # Fazer predição
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        # Todas as probabilidades
        probabilities = [float(p) for p in predictions[0]]

        return jsonify({
            "digit": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    """Verifica status do modelo"""
    return jsonify({
        "model_loaded": model is not None,
        "tensorflow_version": tf.__version__
    })


if __name__ == "__main__":
    print("=" * 50)
    print("TensorFlow POC - Interface Web")
    print("=" * 50)

    # Carregar ou treinar modelo
    if not load_model():
        print("Modelo não encontrado. Treinando...")
        train_model()

    print("\nIniciando servidor web...")
    print("Acesse: http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host="0.0.0.0", port=5000)
