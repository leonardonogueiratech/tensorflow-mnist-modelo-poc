# TensorFlow POC - Classificação de Dígitos MNIST

Proof of Concept demonstrando o uso do TensorFlow para criar uma rede neural convolucional (CNN) que classifica dígitos manuscritos.

## Estrutura do Projeto

```
TensorFlow/
├── src/
│   ├── train.py      # Script de treinamento
│   └── predict.py    # Script de predição/inferência
├── models/           # Modelos treinados e visualizações
├── data/             # Dados (opcional)
├── requirements.txt  # Dependências
└── README.md
```

## Requisitos

- Python 3.9+
- TensorFlow 2.15+

## Instalação

1. Criar ambiente virtual (recomendado):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

2. Instalar dependências:

```bash
pip install -r requirements.txt
```

## Uso

### Treinar o Modelo

```bash
python src/train.py
```

O script irá:
- Baixar automaticamente o dataset MNIST
- Treinar uma CNN por 10 épocas
- Salvar o modelo em `models/mnist_model.keras`
- Gerar gráficos de treinamento

### Fazer Predições

Demonstração interativa (usa imagens do dataset):
```bash
python src/predict.py
```

Demonstração visual com gráfico:
```bash
python src/predict.py --demo
```

Classificar uma imagem personalizada:
```bash
python src/predict.py --image caminho/para/imagem.png
```

### Interface Web

Execute a aplicação Flask:
```bash
python app.py
```

Acesse http://localhost:5000 e desenhe um número para a IA reconhecer!

## Arquitetura do Modelo

```
CNN Sequential:
├── Conv2D (32 filtros, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Conv2D (64 filtros, 3x3) + ReLU
├── MaxPooling2D (2x2)
├── Flatten
├── Dropout (0.5)
├── Dense (128) + ReLU
└── Dense (10) + Softmax
```

## Resultados Esperados

- Acurácia no conjunto de teste: ~99%
- Tempo de treinamento: ~2-5 minutos (CPU) / ~30 segundos (GPU)

## Tecnologias

- **TensorFlow/Keras**: Framework de deep learning
- **NumPy**: Manipulação de arrays
- **Matplotlib**: Visualização de dados
- **Pillow**: Processamento de imagens
- **Flask**: Interface web
