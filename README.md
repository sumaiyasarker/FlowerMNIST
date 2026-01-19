# FlowerMNIST 🌸

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.15-red)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FL-purple)](https://flower.dev/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**FlowerMNIST** is a federated learning project using **Flower** and **PyTorch** to train a CNN on the **MNIST dataset** across multiple simulated clients. This demonstrates how models can learn collaboratively **without sharing raw data**.

---

## 🚀 Features

- Simulates **10 federated clients** using Flower.
- Trains a **Convolutional Neural Network (CNN)** for digit classification.
- Implements **FedAvg** (federated averaging) strategy.
- Tracks **global accuracy** across federated rounds.
- Compatible with **CPU and GPU**.
- Visualizes performance via plots.

---

## 🧩 Project Structure

1.flwr_fed_mnist.py # Main federated learning script
2.README.md # Project documentation

The script will:

1.Split MNIST data across simulated clients.

2.Train a CNN on each client for one epoch per round.

3.Aggregate models using FedAvg.

4.Evaluate accuracy and loss globally.

5.Plot global accuracy over federated rounds.

🧠 How It Works

Data Partitioning: MNIST training data is split among NUM_CLIENTS.

Client Training: Each client trains a CNN locally.

Federated Averaging: Server aggregates client models into a global model.

Evaluation: Accuracy is computed centrally for each round.

📊 Example Output

A plot showing global accuracy across rounds:

⚙️ Configuration

NUM_CLIENTS – Number of federated clients (default: 10)

num_rounds – Number of federated learning rounds (default: 5)

batch_size – Training batch size (default: 32)

DEVICE – cuda if GPU is available, otherwise cpu

You can modify these parameters directly in flwr_fed_mnist.py.

📚 References

Flower: A Friendly Federated Learning Framework

PyTorch Documentation

MNIST Dataset






