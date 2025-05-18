# SAFE-PFL: Efficient and Secure Personalized Federated Learning



This repository contains the implementation of a privacy-preserving federated learning system using homomorphic encryption.

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd safe-pfl-he
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The project uses a YAML configuration file (`config.yaml`) to set various parameters. Key configuration options include:

- `device`: 'cuda' or 'cpu'
- `gpu_index`: GPU device index to use
- `model_type`: Type of model to use (e.g., "cnn", "resnet18", etc.)
- `dataset_type`: Dataset to use (e.g., "fmnist", "cifar10", etc.)
- `federated_learning_schema`: Training schema to use
- `number_of_clients`: Number of federated learning clients
- `federated_learning_rounds`: Number of training rounds
- `encryption_method`: Encryption method to use (e.g., "he_xmkckks")

## Running the Project

1. Configure your experiment by modifying `config.yaml`:
```yaml
device: 'cuda'
gpu_index: "0"
model_type: "cnn"
dataset_type: "fmnist"
# ... other configuration options
```

2. Run the main script:
```bash
python main.py --config-yaml-path ./config.yaml
```

Alternatively, you can use the Jupyter notebook:
```bash
jupyter notebook safe-pfl.ipynb # (just includes learning logic)
```

## Available Models

- CNN
- ResNet18
- ResNet50
- MobileNetV2
- AlexNet
- VGG16
- Vision Transformer (ViT)

## Supported Datasets

- MNIST
- Fashion MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- STL-10
- TinyImageNet

## Project Structure

- `main.py`: Entry point of the application
- `config.yaml`: Configuration file
- `core/`: Core implementation of federated learning components
  - `client.py`: Client implementation
  - `server.py`: Server implementation
- `utils/`: Utility functions and helpers
- `nets/`: Neural network model implementations
- `data/`: Data loading and preprocessing utilities

## Citation

If you use this code in your research, please cite:

```bibtex
  not yet!
```