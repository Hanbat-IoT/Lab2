"""
BWA (Bandwidth Allocation) Algorithm
DRL-Based Dynamic Batch Size Optimization for Federated Learning

Based on Algorithm 1: DRL-Based BWA Algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ActorNetwork(nn.Module):
    """
    Policy network π_θ_a(a^t|s^t) that outputs action (batch size)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # Output probability distribution over actions
        )
    
    def forward(self, state):
        """
        Input: state s^t
        Output: action probability distribution
        """
        return self.network(state)


class CriticNetwork(nn.Module):
    """
    Value network V_θ_v(s) that estimates state value
    """
    def __init__(self, state_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output single value
        )
    
    def forward(self, state):
        """
        Input: state s^t
        Output: state value V(s)
        """
        return self.network(state)


class ExperienceBuffer:
    """
    Experience buffer D for storing (s^t, a^t, s^{t+1}, r^t)
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, next_state, reward):
        """Store experience tuple"""
        self.buffer.append((state, action, next_state, reward))
    
    def sample(self, batch_size):
        """Sample random batch from buffer"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, next_states, rewards = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(next_states),
            torch.FloatTensor(rewards)
        )
    
    def clear(self):
        """Clear the experience buffer"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)


class BWAAlgorithm:
    """
    DRL-Based BWA Algorithm for dynamic batch size optimization
    """
    def __init__(
        self,
        num_clients,
        batch_size_options=[16, 32, 64, 128, 256],
        state_dim=None,
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-3,
        gamma=0.99,
        buffer_capacity=10000,
        ppo_epochs=10,
        ppo_clip=0.2
    ):
        """
        Args:
            num_clients: Number of clients K
            batch_size_options: Available batch sizes
            state_dim: Dimension of state space
            learning_rate_actor: Learning rate for actor network
            learning_rate_critic: Learning rate for critic network
            gamma: Discount factor γ
            buffer_capacity: Capacity of experience buffer
            ppo_epochs: Number of PPO update epochs M
            ppo_clip: PPO clipping parameter ε
        """
        self.num_clients = num_clients
        self.batch_size_options = batch_size_options
        self.action_dim = len(batch_size_options)
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip
        
        # State dimension: [loss, accuracy, round_time] + [data_distribution per client]
        if state_dim is None:
            state_dim = 3 + num_clients  # 3 global metrics + per-client data distribution
        self.state_dim = state_dim
        
        # Initialize Actor network θ_a
        self.actor = ActorNetwork(state_dim, self.action_dim)
        self.actor_old = ActorNetwork(state_dim, self.action_dim)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        
        # Initialize Critic network θ_v
        self.critic = CriticNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        
        # Initialize experience buffer D
        self.experience_buffer = ExperienceBuffer(buffer_capacity)
        
        # θ_a^old ← θ_a (line 3)
        self.theta_a_old = None
        self.update_old_policy()
        
        print(f"BWA Algorithm initialized:")
        print(f"  Clients: {num_clients}")
        print(f"  Batch size options: {batch_size_options}")
        print(f"  State dimension: {state_dim}")
        print(f"  Action dimension: {self.action_dim}")
    
    def update_old_policy(self):
        """θ_a^old ← θ_a (line 22)"""
        self.actor_old.load_state_dict(self.actor.state_dict())
    
    def get_action(self, state, deterministic=False):
        """
        Input s^t to policy network π_θ_a^old(a^t|s^t) to derive action a^t (line 7)
        
        Args:
            state: Current state s^t
            deterministic: If True, select argmax; else sample from distribution
        
        Returns:
            action_idx: Index of selected action
            batch_size: Corresponding batch size b^t
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.actor_old(state_tensor)
        
        if deterministic:
            action_idx = torch.argmax(action_probs).item()
        else:
            # Sample action from probability distribution
            action_idx = torch.multinomial(action_probs, 1).item()
        
        batch_size = self.batch_size_options[action_idx]
        
        return action_idx, batch_size
    
    def calculate_reward(self, loss_improvement, accuracy_improvement, time_cost, lambda_k):
        """
        Calculate reward according to Eq. (8) (line 13)
        
        Reward function balances:
        - Model performance improvement (loss, accuracy)
        - Training efficiency (time cost)
        - Client contribution (λ_k)
        
        Args:
            loss_improvement: Decrease in loss
            accuracy_improvement: Increase in accuracy
            time_cost: Time taken for training
            lambda_k: Client weight/contribution
        
        Returns:
            reward: Calculated reward r^t
        """
        # Reward = α × accuracy_improvement - β × time_cost + γ × loss_improvement
        # Weighted by client contribution λ_k
        
        alpha = 1.0  # Weight for accuracy
        beta = 0.1   # Weight for time cost
        gamma_reward = 0.5  # Weight for loss improvement
        
        reward = (
            alpha * accuracy_improvement +
            gamma_reward * loss_improvement -
            beta * time_cost
        ) * lambda_k
        
        return reward
    
    def store_experience(self, state, action, next_state, reward):
        """
        Store experience (s^t, a^t, s^{t+1}, r^t) in D (line 14)
        """
        self.experience_buffer.push(state, action, next_state, reward)
    
    def update_actor_ppo(self, states, actions, rewards, next_states):
        """
        Update actor network θ_a with experience data using PPO (line 17)
        
        PPO (Proximal Policy Optimization) ensures stable policy updates
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values
            advantages = td_targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action probabilities
        with torch.no_grad():
            old_action_probs = self.actor_old(states)
            old_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get current action probabilities
            action_probs = self.actor(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)) + 1e-8)
            
            # Calculate ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # PPO clipped objective
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages.unsqueeze(1)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        return actor_loss.item()
    
    def update_critic(self, states, rewards, next_states):
        """
        Update critic network θ_v by minimizing loss function (line 18-19)
        
        Loss function: F(θ_v) = Σ(r_i + γ V_θ_v(s_{i+1}) - V_θ_v(s_i))^2
        """
        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        
        # Calculate TD targets
        with torch.no_grad():
            next_values = self.critic(next_states).squeeze()
            td_targets = rewards + self.gamma * next_values
        
        # Current value estimates
        values = self.critic(states).squeeze()
        
        # MSE loss
        critic_loss = nn.MSELoss()(values, td_targets)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return critic_loss.item()
    
    def train_step(self, batch_size=64):
        """
        Perform one training step (lines 16-20)
        
        for m = 1,2,...,M do:
            - Update actor network with PPO
            - Update critic network by minimizing loss
        """
        if len(self.experience_buffer) < batch_size:
            return None, None
        
        # Sample experience data
        states, actions, next_states, rewards = self.experience_buffer.sample(batch_size)
        
        # Update actor with PPO (line 17)
        actor_loss = self.update_actor_ppo(
            states.numpy(),
            actions.numpy(),
            rewards.numpy(),
            next_states.numpy()
        )
        
        # Update critic (line 18-19)
        critic_loss = self.update_critic(
            states.numpy(),
            rewards.numpy(),
            next_states.numpy()
        )
        
        return actor_loss, critic_loss
    
    def clear_buffer(self):
        """Clear experience buffer D (line 23)"""
        self.experience_buffer.clear()
    
    def save_models(self, path_prefix="bwa"):
        """Save actor and critic networks"""
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic.pth")
        print(f"Models saved: {path_prefix}_actor.pth, {path_prefix}_critic.pth")
    
    def load_models(self, path_prefix="bwa"):
        """Load actor and critic networks"""
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic.pth"))
        self.update_old_policy()
        print(f"Models loaded: {path_prefix}_actor.pth, {path_prefix}_critic.pth")


def create_state(loss, accuracy, data_distribution, round_time):
    """
    Create state s^t = [{L^{t+1}(w)}, {φ^{t+1}}, ||d^{t+1}||] (line 12)
    
    Args:
        loss: Current loss L^{t+1}(w)
        accuracy: Current accuracy φ^{t+1}
        data_distribution: Data distribution ||d^{t+1}||
        round_time: Round time
    
    Returns:
        state: State vector
    """
    state = [loss, accuracy, round_time]
    
    # Add data distribution for each client
    if isinstance(data_distribution, (list, np.ndarray)):
        state.extend(data_distribution)
    else:
        state.append(data_distribution)
    
    return np.array(state, dtype=np.float32)


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("BWA Algorithm - Example Usage")
    print("=" * 70)
    
    # Initialize BWA
    num_clients = 3
    bwa = BWAAlgorithm(
        num_clients=num_clients,
        batch_size_options=[16, 32, 64, 128],
        learning_rate_actor=1e-4,
        learning_rate_critic=1e-3,
        gamma=0.99,
        ppo_epochs=10
    )
    
    # Simulate one round
    print("\n--- Simulating Round 1 ---")
    
    # Create initial state
    state = create_state(
        loss=0.5,
        accuracy=0.75,
        data_distribution=[0.3, 0.4, 0.3],
        round_time=10.5
    )
    print(f"State: {state}")
    
    # Get action (batch size)
    action_idx, batch_size = bwa.get_action(state)
    print(f"Selected batch size: {batch_size}")
    
    # Simulate training and get next state
    next_state = create_state(
        loss=0.45,
        accuracy=0.78,
        data_distribution=[0.3, 0.4, 0.3],
        round_time=9.8
    )
    
    # Calculate reward
    reward = bwa.calculate_reward(
        loss_improvement=0.05,
        accuracy_improvement=0.03,
        time_cost=9.8,
        lambda_k=1.0
    )
    print(f"Reward: {reward:.4f}")
    
    # Store experience
    bwa.store_experience(state, action_idx, next_state, reward)
    print(f"Experience buffer size: {len(bwa.experience_buffer)}")
    
    # Train (after collecting enough experiences)
    print("\n--- Training BWA Networks ---")
    for i in range(100):
        # Collect more experiences
        for _ in range(10):
            s = np.random.randn(bwa.state_dim)
            a, _ = bwa.get_action(s)
            s_next = np.random.randn(bwa.state_dim)
            r = np.random.randn()
            bwa.store_experience(s, a, s_next, r)
        
        # Train
        actor_loss, critic_loss = bwa.train_step(batch_size=32)
        
        if actor_loss is not None and i % 20 == 0:
            print(f"Iteration {i}: Actor Loss = {actor_loss:.4f}, Critic Loss = {critic_loss:.4f}")
    
    # Update old policy
    bwa.update_old_policy()
    print("\nOld policy updated (θ_a^old ← θ_a)")
    
    # Clear buffer
    bwa.clear_buffer()
    print("Experience buffer cleared")
    
    print("\n" + "=" * 70)
    print("✅ BWA Algorithm example completed!")
    print("=" * 70)
