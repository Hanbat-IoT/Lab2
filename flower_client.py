"""
Federated Learning Client using Flower Framework

example:
  python flower_client.py --client_id 0 --server_address 192.168.0.100:8080 --dataset mnist
"""

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


class FlowerClient(fl.client.Client):

    def __init__(self, client_id: int, dataset: str, data_size: int = 2500, iid: bool = True):
        self.client_id = client_id
        self.dataset = dataset
        self.data_size = data_size
        self.iid = iid

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

        logging.info(f"Client {client_id} initialized")
        logging.info(f"Device: {self.device}")
        logging.info(f"Dataset: {dataset}, Size: {len(self.data)}")

    def _load_data(self):
        """Load dataset"""
        generator = utils.get_data(self.dataset)
        generator.load_data()
        data = generator.generate()
        labels = generator.labels

        # Create data loader
        # IID=True: 모든 클래스 골고루 분포
        # IID=False: Non-IID, 클라이언트마다 특정 클래스 편향
        loader = utils.Loader(
            argparse.Namespace(
                dataset=self.dataset,
                num_clients=1,
                IID=self.iid,  # 명령줄 인자로 제어
                seed=42
            ),
            generator
        )

        # Get partition for this client
        self.data = loader.get_partition(self.data_size)
        self.full_data = self.data.copy()  # Keep full dataset

        # Test set
        self.testset = loader.get_testset()

        logging.info(f"Loaded {len(self.data)} training samples")

    def get_parameters(self):
        """Return model parameters (Flower 0.18.0 API - no config parameter)"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters (Flower 0.18.0 API)"""
        # Convert Parameters object to list of numpy arrays
        if hasattr(parameters, 'tensors'):
            # Flower 0.18.0: Parameters object has 'tensors' attribute
            weights = fl.common.parameters_to_weights(parameters)
        else:
            # Already a list of numpy arrays
            weights = parameters
            
        params_dict = zip(self.model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, ins):
        """
        Train model with ADM-adjusted data size (Flower 0.18.0 API)

        ins: FitIns object containing parameters and config
        """
        # Extract parameters and config from FitIns
        parameters = ins.parameters
        config = ins.config
        
        self.set_parameters(parameters)

        # Get training config
        v_n = config.get("v_n", 1.0)
        local_epochs = config.get("local_epochs", 5)
        batch_size = config.get("batch_size", 32)
        server_round = config.get("server_round", 0)

        logging.info(f"\n{'='*50}")
        logging.info(f"Round {server_round} - Client {self.client_id}")
        logging.info(f"v_n = {v_n:.3f} (using {v_n*100:.1f}% of data)")
        logging.info(f"{'='*50}")

        # Apply ADM: Adjust data size based on v_n
        adjusted_data = self._apply_adm(v_n)
        logging.info(f"Training samples: {len(adjusted_data)} / {len(self.full_data)}")

        # Train
        start_time = time.time()
        self._train(adjusted_data, local_epochs, batch_size)
        training_time = time.time() - start_time

        logging.info(f"Training completed in {training_time:.2f}s")

        # Return FitRes (Flower 0.18.0 API)
        # Convert parameters to Parameters object
        parameters_obj = fl.common.weights_to_parameters(self.get_parameters())
        
        return fl.common.FitRes(
            parameters=parameters_obj,
            num_examples=len(adjusted_data),
            metrics={
                "client_id": self.client_id,
                "training_time": training_time,
                "v_n": v_n,
                "samples_used": len(adjusted_data)
            }
        )

    def _apply_adm(self, v_n: float):
        """
        Apply ADM Algorithm 1: Data Selection
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
            f"Data distribution after ADM: "
            f"{[len([x for x in selected_data if x[1] == i]) for i in range(10)]}"
        )

        return selected_data

    def _train(self, data, epochs, batch_size):
        """Train the model"""
        self.model.train()

        # Create dataloader
        trainloader = DataLoader(data, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)  # 0.01 → 0.001
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
            logging.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def evaluate(self, ins):
        """Evaluate the model (Flower 0.18.0 API)"""
        # Extract parameters from EvaluateIns
        parameters = ins.parameters
        
        self.set_parameters(parameters)

        testloader = updateModel.get_testloader(self.testset, batch_size=1000)
        accuracy = updateModel.test(self.model, testloader)

        logging.info(f"Client {self.client_id} - Test Accuracy: {100*accuracy:.2f}%")

        # Return EvaluateRes (Flower 0.18.0 API)
        return fl.common.EvaluateRes(
            loss=float(0.0),
            num_examples=len(self.testset),
            metrics={"accuracy": float(accuracy)}
        )


def main():
    parser = argparse.ArgumentParser(description='Flower FL Client')
    parser.add_argument('--client_id', type=int, required=True,
                       help='Client ID (0, 1, 2, ...)')
    parser.add_argument('--server_address', type=str, default='localhost:8080',
                       help='Server address (IP:PORT)')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar'],
                       help='Dataset to use')
    parser.add_argument('--data_size', type=int, default=2500,
                       help='Number of training samples per client')
    parser.add_argument('--iid', type=bool, default=True,
                       help='IID (True) or Non-IID (False) data distribution')

    args = parser.parse_args()

    print("=" * 70)
    print(f"Federated Learning Client {args.client_id}")
    print("=" * 70)
    print(f"Server: {args.server_address}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data size: {args.data_size}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 70)

    # Create client
    client = FlowerClient(
        client_id=args.client_id,
        dataset=args.dataset,
        data_size=args.data_size,
        iid=args.iid
    )

    # Start client (Flower 0.18.0 API)
    fl.client.start_client(
        server_address=args.server_address,
        client=client,
    )


if __name__ == "__main__":
    main()
