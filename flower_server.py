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
from BWA import BWAAlgorithm, create_state
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
        self.round_start_time = None
        
        # Client performance tracking
        self.client_training_times = {}  # {client_id: [time1, time2, ...]}
        self.calibrated = False  # ADM 파라미터 보정 여부

        logging.info("Initialized FedAvgADMStrategy with ADM optimization")
        logging.info("ADM will calibrate based on actual client performance")

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
        self.round_start_time = time.time()  # 라운드 시작 시간 기록

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

    def _calibrate_adm_parameters(self):
        """
        실제 학습 시간을 기반으로 ADM 파라미터 보정
        """
        logging.info("\n[ADM Calibration] Adjusting parameters based on actual performance")
        
        if not self.client_training_times:
            logging.warning("No training time data available for calibration")
            return
        
        # 각 클라이언트의 평균 학습 시간 계산
        avg_times = {}
        for client_id, times in self.client_training_times.items():
            avg_times[client_id] = np.mean(times)
        
        # 시간 기준으로 정렬 (빠른 순)
        sorted_clients = sorted(avg_times.items(), key=lambda x: x[1])
        
        logging.info("Client performance ranking (fastest to slowest):")
        for client_id, avg_time in sorted_clients:
            logging.info(f"  Client {client_id}: {avg_time:.2f}s")
        
        # Frequency 역산
        # 가장 빠른 클라이언트를 3.0 GHz로 설정하고, 시간 비율로 다른 클라이언트 계산
        fastest_time = sorted_clients[0][1]
        
        for client_id, avg_time in sorted_clients:
            # frequency는 시간에 반비례
            # 빠른 클라이언트 (짧은 시간) → 높은 frequency
            relative_speed = fastest_time / avg_time
            frequency_ghz = 3.0 * relative_speed  # 최대 3.0 GHz
            frequency_ghz = max(1.0, min(3.5, frequency_ghz))  # 1.0 ~ 3.5 GHz 범위
            
            if self.parameters is not None:
                self.parameters["frequency_n"][client_id] = frequency_ghz * 1e9
            
            logging.info(f"  Client {client_id}: Calibrated frequency = {frequency_ghz:.2f} GHz")
        
        # ADM 파라미터 업데이트
        if self.parameters is not None:
            # 실제 시간 기반으로 t 값도 조정
            max_time = max(avg_times.values())
            # t는 가장 느린 클라이언트가 v_n=1.0으로 학습할 수 있는 시간
            # 여유를 두기 위해 1.2배
            self.parameters["t"] = max_time * 1.2
            logging.info(f"  Adjusted t = {self.parameters['t']:.2f}s (based on slowest client)")
        
        logging.info("[ADM Calibration] Completed\n")

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

        # Log client training info and collect performance data
        total_samples = 0
        for _, fit_res in results:
            num_samples = fit_res.num_examples
            training_time = fit_res.metrics.get("training_time", 0)
            client_id = fit_res.metrics.get("client_id", -1)
            total_samples += num_samples

            # Collect training times for calibration
            if client_id not in self.client_training_times:
                self.client_training_times[client_id] = []
            self.client_training_times[client_id].append(training_time)

            logging.info(
                f"Client {client_id}: {num_samples} samples, "
                f"training time: {training_time:.2f}s"
            )
        
        # Calibrate ADM parameters after first round
        if rnd == 1 and not self.calibrated:
            self._calibrate_adm_parameters()
            self.calibrated = True

        # Call parent's aggregate_fit
        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        
        # Manually evaluate on server (Flower 0.18.0 workaround)
        # Only evaluate once per round (not per client)
        if aggregated_params and len(self.accuracies) < rnd:
            accuracy = self._evaluate_global_model(aggregated_params)
            self.accuracies.append(accuracy)
            
            # 라운드 시간 기록
            if self.round_start_time is not None:
                round_time = time.time() - self.round_start_time
                self.round_times.append(round_time)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
            if self.round_times:
                logging.info(f"Round Time: {self.round_times[-1]:.2f}s")
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

        # Don't append here - already done in aggregate_fit
        logging.info(f"Client-side evaluation: {100 * accuracy:.2f}%")

        return accuracy, {"accuracy": accuracy}

    def save_results(self, filename: str):
        """Save experiment results"""
        results = {
            "strategy": "FedAvg+ADM",
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": len(self.accuracies),
            "accuracies": self.accuracies,
            "round_times": self.round_times,
            "total_time": sum(self.round_times) if self.round_times else 0,
            "avg_round_time": sum(self.round_times) / len(self.round_times) if self.round_times else 0,
            "v_n_history": self.v_n_history,
            "adm_params": self.adm_params
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {filename}")
        if self.round_times:
            logging.info(f"Total training time: {sum(self.round_times):.2f}s")
            logging.info(f"Average round time: {sum(self.round_times)/len(self.round_times):.2f}s")


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
        self.round_times = []
        self.round_start_time = None

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
            
            # 라운드 시간 기록
            if self.round_start_time is not None:
                round_time = time.time() - self.round_start_time
                self.round_times.append(round_time)
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
            if self.round_times:
                logging.info(f"Round Time: {self.round_times[-1]:.2f}s")
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

        self.round_start_time = time.time()  # 라운드 시작 시간 기록

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

        # Don't append here - already done in aggregate_fit
        logging.info(f"Client-side evaluation: {100 * accuracy:.2f}%")

        return accuracy, {"accuracy": accuracy}

    def save_results(self, filename: str):
        """Save experiment results"""
        results = {
            "strategy": "FedAvg (Baseline)",
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": len(self.accuracies),
            "accuracies": self.accuracies,
            "round_times": self.round_times,
            "total_time": sum(self.round_times) if self.round_times else 0,
            "avg_round_time": sum(self.round_times) / len(self.round_times) if self.round_times else 0
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

        logging.info(f"Results saved to {filename}")
        if self.round_times:
            logging.info(f"Total training time: {sum(self.round_times):.2f}s")
            logging.info(f"Average round time: {sum(self.round_times)/len(self.round_times):.2f}s")


class FedAvgBWAStrategy(FedAvg):
    """
    FedAvg + BWA Strategy
    DRL 기반 동적 배치 크기 최적화
    """
    def __init__(
        self,
        num_clients: int,
        dataset: str,
        num_rounds: int,
        batch_size_options=[16, 32, 64, 128],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_clients = num_clients
        self.dataset = dataset
        self.num_rounds = num_rounds
        self.batch_size_options = batch_size_options
        
        # Performance tracking
        self.accuracies = []
        self.round_times = []
        self.batch_size_history = []
        self.round_start_time = None
        
        # Previous metrics for reward calculation
        self.prev_loss = None
        self.prev_accuracy = None
        
        # Initialize BWA algorithm
        # State dimension: [loss, accuracy, round_time] + [data_dist per client]
        state_dim = 3 + num_clients
        self.bwa = BWAAlgorithm(
            num_clients=num_clients,
            batch_size_options=batch_size_options,
            state_dim=state_dim,
            learning_rate_actor=1e-4,
            learning_rate_critic=1e-3,
            gamma=0.99,
            ppo_epochs=10
        )
        
        logging.info(f"BWA state dimension: {state_dim} (3 metrics + {num_clients} clients)")
        
        logging.info("Initialized FedAvg+BWA Strategy with DRL-based batch size optimization")

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters"""
        model = get_model(self.dataset)
        params = [val.cpu().numpy() for _, val in model.state_dict().items()]
        return fl.common.weights_to_parameters(params)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        """Configure clients with BWA-optimized batch sizes"""
        
        self.round_start_time = time.time()
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Round {rnd}/{self.num_rounds}")
        logging.info(f"{'='*60}")
        
        # Create state for BWA
        if self.prev_accuracy is not None:
            # Calculate data distribution (simplified)
            data_distribution = [1.0 / self.num_clients] * self.num_clients
            
            state = create_state(
                loss=self.prev_loss if self.prev_loss else 1.0,
                accuracy=self.prev_accuracy if self.prev_accuracy else 0.0,
                data_distribution=data_distribution,
                round_time=self.round_times[-1] if self.round_times else 10.0
            )
            
            # Get action (batch size) from BWA
            action_idx, batch_size = self.bwa.get_action(state)
            logging.info(f"\n[BWA] Selected batch size: {batch_size}")
        else:
            # First round: use default batch size
            batch_size = 32
            action_idx = self.batch_size_options.index(batch_size) if batch_size in self.batch_size_options else 1
            logging.info(f"\n[BWA] First round - using default batch size: {batch_size}")
        
        self.batch_size_history.append(batch_size)
        
        # Get clients
        clients = client_manager.sample(
            num_clients=self.num_clients,
            min_num_clients=self.num_clients
        )
        
        # Configure each client with BWA-optimized batch size
        config_list = []
        for idx, client in enumerate(clients):
            config = {
                "server_round": rnd,
                "local_epochs": 5,
                "batch_size": batch_size,  # BWA optimized
                "v_n": 1.0,
                "client_id": idx
            }
            fit_ins = fl.common.FitIns(parameters, config)
            config_list.append((client, fit_ins))
        
        logging.info(f"All clients: batch_size = {batch_size} (BWA optimized)")
        
        return config_list

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates and train BWA"""
        
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
        
        # Aggregate parameters
        aggregated_params, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        
        # Evaluate and update BWA
        if aggregated_params and len(self.accuracies) < rnd:
            # Evaluate global model
            accuracy = self._evaluate_global_model(aggregated_params)
            loss = 1.0 - accuracy  # Simplified loss
            
            self.accuracies.append(accuracy)
            
            # Record round time
            if self.round_start_time is not None:
                round_time = time.time() - self.round_start_time
                self.round_times.append(round_time)
            
            # BWA: Calculate reward and store experience
            if self.prev_accuracy is not None:
                # Calculate improvements
                accuracy_improvement = accuracy - self.prev_accuracy
                loss_improvement = self.prev_loss - loss
                time_cost = self.round_times[-1] if self.round_times else 10.0
                
                # Calculate reward
                reward = self.bwa.calculate_reward(
                    loss_improvement=loss_improvement,
                    accuracy_improvement=accuracy_improvement,
                    time_cost=time_cost,
                    lambda_k=1.0
                )
                
                # Create states
                data_distribution = [1.0 / self.num_clients] * self.num_clients
                prev_state = create_state(
                    self.prev_loss, self.prev_accuracy,
                    data_distribution,
                    self.round_times[-2] if len(self.round_times) > 1 else 10.0
                )
                curr_state = create_state(
                    loss, accuracy,
                    data_distribution,
                    time_cost
                )
                
                # Store experience
                action_idx = self.batch_size_options.index(self.batch_size_history[-1])
                self.bwa.store_experience(prev_state, action_idx, curr_state, reward)
                
                logging.info(f"[BWA] Reward: {reward:.4f}, Buffer size: {len(self.bwa.experience_buffer)}")
                
                # Train BWA networks
                if len(self.bwa.experience_buffer) >= 32:
                    actor_loss, critic_loss = self.bwa.train_step(batch_size=32)
                    if actor_loss is not None:
                        logging.info(f"[BWA] Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            
            # Update previous metrics
            self.prev_loss = loss
            self.prev_accuracy = accuracy
            
            logging.info(f"\n{'='*60}")
            logging.info(f"Round {rnd} - Global Accuracy: {100 * accuracy:.2f}%")
            if self.round_times:
                logging.info(f"Round Time: {self.round_times[-1]:.2f}s")
            logging.info(f"{'='*60}\n")
        
        return aggregated_params, metrics

    def _evaluate_global_model(self, parameters):
        """Evaluate global model on test set"""
        generator = utils.get_data(self.dataset)
        generator.load_data()
        testset = generator.testset
        
        weights = fl.common.parameters_to_weights(parameters)
        model = get_model(self.dataset)
        
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        testloader = updateModel.get_testloader(testset, batch_size=1000)
        accuracy = updateModel.test(model, testloader)
        
        return accuracy

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results"""
        
        if not results:
            return None, {}
        
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        accuracy = sum(accuracies) / sum(examples) if sum(examples) > 0 else 0
        
        logging.info(f"Client-side evaluation: {100 * accuracy:.2f}%")
        
        return accuracy, {"accuracy": accuracy}

    def save_results(self, filename: str):
        """Save experiment results"""
        results = {
            "strategy": "FedAvg+BWA",
            "dataset": self.dataset,
            "num_clients": self.num_clients,
            "num_rounds": len(self.accuracies),
            "accuracies": self.accuracies,
            "round_times": self.round_times,
            "batch_size_history": self.batch_size_history,
            "total_time": sum(self.round_times) if self.round_times else 0,
            "avg_round_time": sum(self.round_times) / len(self.round_times) if self.round_times else 0
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logging.info(f"Results saved to {filename}")
        if self.round_times:
            logging.info(f"Total training time: {sum(self.round_times):.2f}s")
            logging.info(f"Average round time: {sum(self.round_times)/len(self.round_times):.2f}s")
        
        # Save BWA models
        bwa_model_path = filename.replace('.json', '_bwa')
        self.bwa.save_models(bwa_model_path)


def main():
    parser = argparse.ArgumentParser(description='Flower FL Server')
    parser.add_argument('--strategy', type=str, default='fedavg_adm',
                       choices=['fedavg', 'fedavg_adm', 'fedavg_bwa'],
                       help='Strategy: fedavg (baseline), fedavg_adm, or fedavg_bwa')
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
            fraction_fit=1.0,  # 100% 클라이언트 선택
            fraction_evaluate=1.0,  # 100% 클라이언트 평가
            min_fit_clients=args.num_clients,  # 최소 N개 필요
            min_evaluate_clients=args.num_clients,  # 최소 N개 필요
            min_available_clients=args.num_clients,  # 최소 N개 연결 필요
            accept_failures=False,  # 실패 허용 안함
        )
        result_file = f"results_{args.strategy}_{args.dataset}_{args.num_clients}clients_{timestamp}.json"
    elif args.strategy == 'fedavg_bwa':
        strategy = FedAvgBWAStrategy(
            num_clients=args.num_clients,
            dataset=args.dataset,
            num_rounds=args.num_rounds,
            batch_size_options=[16, 32, 64, 128],
            fraction_fit=1.0,  # 100% 클라이언트 선택
            fraction_evaluate=1.0,  # 100% 클라이언트 평가
            min_fit_clients=args.num_clients,  # 최소 N개 필요
            min_evaluate_clients=args.num_clients,  # 최소 N개 필요
            min_available_clients=args.num_clients,  # 최소 N개 연결 필요
            accept_failures=False,  # 실패 허용 안함
        )
        result_file = f"results_{args.strategy}_{args.dataset}_{args.num_clients}clients_{timestamp}.json"
    else:
        strategy = FedAvgBaselineStrategy(
            num_clients=args.num_clients,
            dataset=args.dataset,
            num_rounds=args.num_rounds,
            fraction_fit=1.0,  # 100% 클라이언트 선택
            fraction_evaluate=1.0,  # 100% 클라이언트 평가
            min_fit_clients=args.num_clients,  # 최소 N개 필요
            min_evaluate_clients=args.num_clients,  # 최소 N개 필요
            min_available_clients=args.num_clients,  # 최소 N개 연결 필요
            accept_failures=False,  # 실패 허용 안함
        )
        result_file = f"results_{args.strategy}_{args.dataset}_{args.num_clients}clients_{timestamp}.json"

    # Start server (Flower 0.18.0 API)
    # Increase timeout to wait for all clients
    fl.server.start_server(
        server_address=args.server_address,
        config={
            "num_rounds": args.num_rounds,
            "round_timeout": 600.0,  # 10분 타임아웃 (느린 클라이언트 대기)
        },
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
