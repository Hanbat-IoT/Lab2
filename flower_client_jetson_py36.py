"""
Federated Learning Client for Jetson Nano (Python 3.6.9)
Flower 1.4.0 compatible version

실행 예시:
  python3 flower_client_jetson_py36.py --client_id 0 --server_address 192.168.0.100:8080 --dataset mnist
"""

import sys
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import logging
import time
import platform
import psutil
from collections import OrderedDict

from models import get_model
import utils
import updateModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[Client][%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)

# Python 3.6 확인
if sys.version_info < (3, 7):
    logging.info("Python 3.6 detected - Using Flower 1.4.0 compatible mode")


class FlowerClient(fl.client.Client):
    """
    Flower Client Implementation for Python 3.6
    Flower 1.4.0 API 사용 (NumPyClient 대신 Client)
    """

    def __init__(self, client_id, dataset, data_size=2500):
        self.client_id = client_id
        self.dataset = dataset
        self.data_size = data_size

        # Load model
        self.model = get_model(dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load and prepare data
        self._load_data()

        # Device info
        self.device_info = {
            'platform': platform.system(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'has_gpu': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }

        logging.info("Client {} initialized".format(client_id))
        logging.info("Device: {}".format(self.device))
        logging.info("Dataset: {}, Size: {}".format(dataset, len(self.data)))

    def _load_data(self):
        """Load dataset"""
        generator = utils.get_data(self.dataset)
        generator.load_data()
        data = generator.generate()
        labels = generator.labels

        # Create data loader
        loader = utils.Loader(
            argparse.Namespace(
                dataset=self.dataset,
                num_clients=1,
                IID=True,
                seed=42
            ),
            generator
        )

        # Get partition for this client
        self.data = loader.get_partition(self.data_size)
        self.full_data = self.data[:]  # Keep full dataset (copy)

        # Test set
        self.testset = loader.get_testset()

        logging.info("Loaded {} training samples".format(len(self.data)))

    def get_parameters(self):
        """Return model parameters (Flower 1.4.0 API - no config param)"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, ins):
        """
        Train model with ADM-adjusted data size (Flower 1.4.0 API)
        
        ins: FitIns object with parameters and config
        """
        # Extract parameters and config
        parameters = ins.parameters
        config = ins.config if hasattr(ins, 'config') else {}
        
        self.set_parameters(parameters)

        # Get training config
        v_n = float(config.get("v_n", 1.0))
        local_epochs = int(config.get("local_epochs", 5))
        batch_size = int(config.get("batch_size", 32))
        server_round = int(config.get("server_round", 0))

        logging.info("\n" + "="*50)
        logging.info("Round {} - Client {}".format(server_round, self.client_id))
        logging.info("v_n = {:.3f} (using {:.1f}% of data)".format(v_n, v_n*100))
        logging.info("="*50)

        # Apply ADM: Adjust data size based on v_n
        adjusted_data = self._apply_adm(v_n)
        logging.info("Training samples: {} / {}".format(len(adjusted_data), len(self.full_data)))

        # Train
        start_time = time.time()
        self._train(adjusted_data, local_epochs, batch_size)
        training_time = time.time() - start_time

        logging.info("Training completed in {:.2f}s".format(training_time))

        # Return FitRes (Flower 1.4.0 format)
        return fl.common.FitRes(
            parameters=self.get_parameters(),
            num_examples=len(adjusted_data),
            metrics={
                "client_id": self.client_id,
                "training_time": training_time,
                "v_n": v_n,
                "samples_used": len(adjusted_data)
            }
        )

    def _apply_adm(self, v_n):
        """
        Apply ADM Algorithm 1: Data Selection
        v_n 값에 따라 각 레이블별로 균등하게 데이터를 선택
        """
        v_n = float(round(v_n, 2))
        target_size = int(len(self.full_data) * v_n)

        if v_n >= 0.99:
            return self.full_data

        # Group data by label
        label_groups = {}
        for item in self.full_data:
            _, label = item
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        # Calculate reduction per label (uniform reduction)
        selected_data = []
        samples_per_label = target_size // len(label_groups)

        for label, items in label_groups.items():
            # Take samples_per_label from each label
            selected = items[:min(samples_per_label, len(items))]
            selected_data.extend(selected)

        # If we need more samples to reach target, add from remaining
        if len(selected_data) < target_size:
            remaining_needed = target_size - len(selected_data)
            all_items = [item for items in label_groups.values() for item in items]
            additional = all_items[:remaining_needed]
            selected_data.extend(additional)

        logging.info(
            "Data distribution after ADM: {}".format(
                [len([x for x in selected_data if x[1] == i]) for i in range(10)]
            )
        )

        return selected_data

    def _train(self, data, epochs, batch_size):
        """Train the model"""
        self.model.train()

        # Create dataloader
        trainloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_data, batch_labels in trainloader:
                batch_data, batch_labels = batch_data.to(self.device), batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(trainloader)
            logging.info("  Epoch {}/{}, Loss: {:.4f}".format(epoch+1, epochs, avg_loss))

    def evaluate(self, ins):
        """Evaluate the model (Flower 1.4.0 API)"""
        parameters = ins.parameters
        config = ins.config if hasattr(ins, 'config') else {}
        
        self.set_parameters(parameters)

        testloader = updateModel.get_testloader(self.testset, batch_size=1000)
        accuracy = updateModel.test(self.model, testloader)

        logging.info("Client {} - Test Accuracy: {:.2f}%".format(self.client_id, 100*accuracy))

        # Return EvaluateRes (Flower 1.4.0 format)
        return fl.common.EvaluateRes(
            loss=float(0.0),
            num_examples=len(self.testset),
            metrics={"accuracy": float(accuracy)}
        )


def main():
    parser = argparse.ArgumentParser(description='Flower FL Client (Jetson Nano Python 3.6)')
    parser.add_argument('--client_id', type=int, required=True,
                       help='Client ID (0, 1, 2, ...)')
    parser.add_argument('--server_address', type=str, default='localhost:8080',
                       help='Server address (IP:PORT)')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar'],
                       help='Dataset to use')
    parser.add_argument('--data_size', type=int, default=1500,
                       help='Number of training samples per client (default: 1500 for Jetson)')

    args = parser.parse_args()

    print("=" * 70)
    print("Federated Learning Client {} (Python 3.6 Compatible)".format(args.client_id))
    print("=" * 70)
    print("Python: {}".format(sys.version))
    print("Flower: {}".format(fl.__version__))
    print("PyTorch: {}".format(torch.__version__))
    print("Server: {}".format(args.server_address))
    print("Dataset: {}".format(args.dataset.upper()))
    print("Data size: {}".format(args.data_size))
    print("Device: {}".format('GPU' if torch.cuda.is_available() else 'CPU'))
    print("=" * 70)

    # Create client
    client = FlowerClient(
        client_id=args.client_id,
        dataset=args.dataset,
        data_size=args.data_size
    )

    # Start client (Flower 1.4.0 API)
    fl.client.start_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == "__main__":
    main()
