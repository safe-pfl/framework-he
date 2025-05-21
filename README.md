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

The project uses a YAML configuration file (`config.yaml`) to set various parameters. Here's a comprehensive list of all available configuration options (italics are not important they are just for sack of documentation):

### Basic Settings
- `device`: Computing device to use ('cuda' or 'cpu')
- `gpu_index`: GPU device index to use (e.g., "0")
- `federation_id`: _Unique identifier for the federation_
- `federated_learning_schema`: _Training schema (e.g., 'TraditionalFederatedLearning')_
- `client_role`: _Role of the client (e.g., 'train')_

### Model and Training Parameters
- `learning_rate`: Learning rate for model training (e.g., 0.001)
- `model_type`: Type of model to use (e.g., "LENET", "CNN", "ResNet18", etc.)
- `pretrained_models`: Whether to use pretrained models (true/false)
- `dataset_type`: Dataset to use (e.g., "mnist", "fmnist", "cifar10", etc.)
- `number_of_epochs`: Number of training epochs per round
- `train_batch_size`: Batch size for training
- `test_batch_size`: Batch size for testing
- `transform_input_size`: Input size for transformations
- `weight_decay`: Weight decay parameter for regularization

### Data Distribution
- `data_distribution_kind`: Type of data distribution (e.g., "20", "30" or "dir")
- `desired_distribution`: Custom distribution settings (null for default)
- `dirichlet_beta`: Beta parameter for Dirichlet distribution

### Federated Learning Settings
- `number_of_clients`: Total number of federated learning clients
- `client_sampling_rate`: _Rate at which clients are sampled (0.0 to 1.0)_
- `federated_learning_rounds`: Number of federated learning rounds
- `stop_avg_accuracy`: Target accuracy to stop training
- `save_before_aggregation_models`: Whether to save models before aggregation
- `save_global_models`: Whether to save global models

### Clustering and Aggregation
- `do_cluster`: _Whether to perform clustering (true/false)_
- `clustering_period`: Period between clustering operations
- `pre_computed_data_driven_clustering`: Whether to use pre-computed clustering
- `aggregation_strategy`: _Strategy for model aggregation (e.g., "FedAvg")_
- `aggregation_sample_scaling`: _Whether to scale samples during aggregation_

### Distance and Sensitivity
- `distance_metric`: Metric for distance calculation (e.g., "coordinate", "cosine", etc.)
- `distance_metric_on_parameters`: Whether to calculate distance on parameters
- `dynamic_sensitivity_percentage`: Whether to use dynamic sensitivity
- `sensitivity_percentage`: Percentage for sensitivity calculation
- `remove_common_ids`: Whether to remove common IDs

### Encryption
- `encryption_method`: Method of encryption (e.g., "he_xmkckks" or null)
- `xmkckks_weight_decimals`: Number of decimal places for encrypted weights (originally 8)

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