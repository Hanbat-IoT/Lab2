"""
Federated Learning Server using Flower Framework
FedAvg vs FedAvg+ADM 비교 실험

실행: python flower_server.py --strategy fedavg_adm --num_rounds 10
"""

import flwr as fl
from flwr.common import Parameters, FitRes, Scalar
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import torch
import logging
import argparse
import json
from collections import OrderedDict
from models import get_model
from ADM import init_param_hetero, block_coordinate_descent
import time
import utils
import updateModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s][%(asctime)s]: %(message)s',
    datefmt='%H:%M:%S'
)


class FedAvgADMStrategy(FedAvg):
    """
    FedAvg + ADM Strategy
    ADM 알고리즘을 사용하여 각 클라이언트의 데이터 사용량(v_n)을 최적화
    """
    def __init__(
        self,
        num_clients: int,
        dataset: str,
        adm_params: Dict,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.dataset = dataset
        self.adm_params = adm_params
        self.current_round = 0

        # ADM parameters
        self.parameters = None
        self.optimal_v_n = [1.0] * num_clients

        # Performance tracking
        self.round_times = []
        self.accuracies = []
        self.v_n_history = []

        logging.info("Initialized FedAvgADMStrategy with ADM optimization")

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters (Flower 0.18.0 API)"""
        model = get_model(self.dataset)

        # Convert model to parameters (Flower 0.18.0 uses weights_to_parameters)
        params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return fl.common.weights_to_parameters(params)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """
        라운드 시작 시 호출: ADM 최적화 수행 및 클라이언트 설정 (Flower 0.18.0 API)
        """
        self.current_round = rnd - 1  # 0-indexed

        logging.info(f"\n{'='*60}")
        logging.info(f"Round {rnd}/{self.adm_params['rounds']}")
        logging.info(f"{'='*60}")

        # ADM configuration
        self._run_adm_optimization()

        # Get all available clients
        sample_size = self.num_clients
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=sample_size
        )

        # Configure each client with its specific v_n
        config_list = []
        for idx, client in enumerate(clients):
            config = {
                "server_round": rnd,
                "local_epochs": 5,
                "batch_size": 32,
                "v_n": float(self.optimal_v_n[idx]),  # ADM optimized value
                "client_id": idx
            }
            fit_ins = fl.common.FitIns(parameters, config)
            config_list.append((client, fit_ins))

            logging.info(f"Client {idx}: v_n = {self.optimal_v_n[idx]:.3f}")

        return config_list

    def _run_adm_optimization(self):
        """ADM 알고리즘 실행"""
        logging.info("\n[ADM Optimization]")

        # Initialize ADM parameters if first round
        if self.parameters is None:
            self.parameters = init_param_hetero(
                self.adm_params,
                self.num_clients,
                self.adm_params['t']
            )

        # Run block coordinate descent
        start_time = time.time()
        self.optimal_v_n, sol_list, optimal_t = block_coordinate_descent(
            self.parameters,
            self.current_round,
            self.parameters["t"]
        )
        adm_time = time.time() - start_time

        # Update parameters
        self.parameters["t"] = optimal_t

        # Round 20마다 v_n 감소 (논문 구현)
        if self.current_round > 0 and self.current_round % 20 == 0:
            self.optimal_v_n = [max(0.4, x - 0.1) for x in self.optimal_v_n]
            logging.info("Applied v_n decay (every 20 rounds)")

        # Store history
        self.v_n_history.append(self.optimal_v_n.copy())

        vn_rounded = [round(num, 3) for num in self.optimal_v_n]
        logging.info(f"Optimized v_n: {vn_rounded}")
        logging.info(f"ADM optimization time: {adm_time:.3f}s")
        logging.info(f"Optimal t: {optimal_t:.6f}s\n")

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates (FedAvg) - Flower 0.18.0 API"""

        if not results:
            return None, {}

        # Log client training info
        total_samples = 0
        for _, fit_res in results:
            num_samples = fit_res.num_examples
            training_time = fit_res.metrics.get("training_time", 0)
            client_id = fit_res.metrics.get("client_id", -1)
            total_samples += num_samples

            logging.info(
                f"Client {client_id}: {num_samples} samples, "
                f"training time: {training_time:.2f}s"
            )

        # Call parent's aggregate_fit
        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        
        # Manually evaluate on server (Flower 0.18.0 workaround)
        # Only evaluate once per round (not per client)
        if aggregated_params and len(self.accuracies) < rnd:
            accuracy = self._evaluate_global_model(aggregated_params)
            self.accuracies.append(accuracy)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
            logging.info(f"{'='*60}\n")

        return aggregated_params, metrics
    
    def _evaluate_global_model(self, parameters):
        """Evaluate global model on test set"""
        import utils
        import updateModel
        
        # Load test data
        generator = utils.get_data(self.dataset)
        generator.load_data()
        testset = generator.testset
        
        # Load parameters into model
        weights = fl.common.parameters_to_weights(parameters)
        model = get_model(self.dataset)
        
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        testloader = updateModel.get_testloader(testset, batch_size=1000)
        accuracy = updateModel.test(model, testloader)
        
        return accuracy

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results (Flower 0.18.0 API)"""

        if not results:
            return None, {}

        # Weighted average of accuracies
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0

        self.accuracies.append(accuracy)

        logging.info(f"\n{'='*60}")
        logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
        logging.info(f"{'='*60}\n")

        return accuracy, {"accuracy": accuracy}

    def save_results(self, filename: str):
        """Save experiment results"""
        results = {
            "strategy": "FedAvg+ADM",
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": len(self.accuracies),
            "accuracies": self.accuracies,
            "v_n_history": self.v_n_history,
            "adm_params": self.adm_params
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {filename}")


class FedAvgBaselineStrategy(FedAvg):
    """
    Baseline FedAvg Strategy (without ADM)
    모든 클라이언트가 항상 v_n = 1.0 (전체 데이터 사용)
    """
    def __init__(self, num_clients: int, dataset: str, num_rounds: int, **kwargs):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.dataset = dataset
        self.num_rounds = num_rounds
        self.accuracies = []

        logging.info("Initialized FedAvg Baseline Strategy (no ADM)")
    
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates and evaluate (Flower 0.18.0 API)"""

        if not results:
            return None, {}

        # Call parent's aggregate_fit
        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        
        # Manually evaluate on server (only once per round)
        if aggregated_params and len(self.accuracies) < rnd:
            accuracy = self._evaluate_global_model(aggregated_params)
            self.accuracies.append(accuracy)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
            logging.info(f"{'='*60}\n")

        return aggregated_params, metrics
    
    def _evaluate_global_model(self, parameters):
        """Evaluate global model on test set"""
        # Load test data
        generator = utils.get_data(self.dataset)
        generator.load_data()
        testset = generator.testset
        
        # Load parameters into model
        weights = fl.common.parameters_to_weights(parameters)
        model = get_model(self.dataset)
        
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        testloader = updateModel.get_testloader(testset, batch_size=1000)
        accuracy = updateModel.test(model, testloader)
        
        return accuracy

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters (Flower 0.18.0 API)"""
        model = get_model(self.dataset)
        params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return fl.common.weights_to_parameters(params)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure clients (all with v_n=1.0) - Flower 0.18.0 API"""

        logging.info(f"\n{'='*60}")
        logging.info(f"Round {rnd}/{self.num_rounds}")
        logging.info(f"{'='*60}")

        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients
        )

        config_list = []
        for idx, client in enumerate(clients):
            config = {
                "server_round": rnd,
                "local_epochs": 5,
                "batch_size": 32,
                "v_n": 1.0,  # Always use full data
                "client_id": idx
            }
            fit_ins = fl.common.FitIns(parameters, config)
            config_list.append((client, fit_ins))

        logging.info("All clients: v_n = 1.0 (baseline)")

        return config_list

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results (Flower 0.18.0 API)"""

        if not results:
            return None, {}

        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0

        self.accuracies.append(accuracy)

        logging.info(f"\n{'='*60}")
        logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
        logging.info(f"{'='*60}\n")

        return accuracy, {"accuracy": accuracy}

    def save_results(self, filename: str):
        """Save experiment results"""
        results = {
            "strategy": "FedAvg (Baseline)",
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": len(self.accuracies),
            "accuracies": self.accuracies
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Flower FL Server')
    parser.add_argument('--strategy', type=str, default='fedavg_adm',
                       choices=['fedavg', 'fedavg_adm'],
                       help='Strategy: fedavg (baseline) or fedavg_adm')
    parser.add_argument('--num_clients', type=int, default=3,
                       help='Number of clients')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of training rounds')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'cifar'],
                       help='Dataset to use')
    parser.add_argument('--server_address', type=str, default='0.0.0.0:8080',
                       help='Server address (IP:PORT)')

    args = parser.parse_args()

    print("=" * 70)
    print("Federated Learning Server - Flower Framework")
    print("=" * 70)
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Clients: {args.num_clients}")
    print(f"Rounds: {args.num_rounds}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Server: {args.server_address}")
    print("=" * 70)
    print("\nWaiting for clients to connect...")
    print("=" * 70)

    # ADM parameters (for FedAvg+ADM strategy)
    adm_params = {
        'sigma': 0.9 * 1e-8,
        'D_n': [2500 for _ in range(args.num_clients)],
        'Gamma': 0.4,
        'local_iter': 10,
        'c_n': 30,
        'frequency_n_GHz': [1.5, 2.0, 2.5, 3.0],  # Heterogeneous devices
        'weight_size_n_kbit': 100,
        'number_of_clients': args.num_clients,
        'bandwidth_MHz': 10,
        'channel_gain_n': 1,
        'transmission_power_n': [0.5, 1.0],
        'noise_W': 10**(-114/10) * 1e-3,
        't': 0.006,
        'rounds': args.num_rounds
    }

    # Generate unique filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select strategy (Flower 0.18.0 API)
    if args.strategy == 'fedavg_adm':
        strategy = FedAvgADMStrategy(
            num_clients=args.num_clients,
            dataset=args.dataset,
            adm_params=adm_params,
            min_available_clients=args.num_clients,
        )
        result_file = f"results_{args.strategy}_{args.dataset}_{args.num_clients}clients_{timestamp}.json"
    else:
        strategy = FedAvgBaselineStrategy(
            num_clients=args.num_clients,
            dataset=args.dataset,
            num_rounds=args.num_rounds,
            min_available_clients=args.num_clients,
        )
        result_file = f"results_{args.strategy}_{args.dataset}_{args.num_clients}clients_{timestamp}.json"

    # Start server (Flower 0.18.0 API)
    fl.server.start_server(
        server_address=args.server_address,
        config={"num_rounds": args.num_rounds},
        strategy=strategy,
    )

    # Save results
    strategy.save_results(result_file)

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Results saved to: {result_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
