"""
Reinforcement Learning Module for Ultimate Trading Bot v2.2

Implements RL algorithms for trading including Q-Learning, DQN, PPO,
and policy gradient methods for adaptive trading strategies.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of trading actions."""
    BUY = 0
    SELL = 1
    HOLD = 2


class RewardType(Enum):
    """Types of reward functions."""
    PROFIT = "profit"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    RISK_ADJUSTED = "risk_adjusted"
    CUSTOM = "custom"


@dataclass
class State:
    """Trading environment state."""
    features: np.ndarray
    position: float
    balance: float
    unrealized_pnl: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_array(self) -> np.ndarray:
        """Convert state to array."""
        return np.concatenate([
            self.features,
            np.array([self.position, self.balance, self.unrealized_pnl])
        ])

    @property
    def shape(self) -> tuple[int, ...]:
        """Get state shape."""
        return self.to_array().shape


@dataclass
class Action:
    """Trading action."""
    action_type: ActionType
    size: float = 1.0
    confidence: float = 1.0

    def to_int(self) -> int:
        """Convert action to integer."""
        return self.action_type.value


@dataclass
class Experience:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Result from an episode."""
    total_reward: float
    total_steps: int
    final_balance: float
    trades: int
    wins: int
    losses: int
    sharpe_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "final_balance": self.final_balance,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "sharpe_ratio": self.sharpe_ratio
        }


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self._buffer: list[Experience] = []
        self._position = 0

        logger.info(f"Initialized ReplayBuffer with capacity={capacity}")

    def push(self, experience: Experience) -> None:
        """
        Add experience to buffer.

        Args:
            experience: Experience to add
        """
        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
        else:
            self._buffer[self._position] = experience

        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Experience]:
        """
        Sample batch of experiences.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of experiences
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        """Get buffer size."""
        return len(self._buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer."""

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self._priorities = np.zeros(capacity)
        self._max_priority = 1.0

    def push(self, experience: Experience) -> None:
        """Add experience with max priority."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
            self._priorities[len(self._buffer) - 1] = self._max_priority
        else:
            self._buffer[self._position] = experience
            self._priorities[self._position] = self._max_priority

        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (experiences, weights, indices)
        """
        buffer_len = len(self._buffer)
        priorities = self._priorities[:buffer_len]

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(buffer_len, size=batch_size, p=probs, replace=False)

        weights = (buffer_len * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        experiences = [self._buffer[i] for i in indices]

        return experiences, weights, indices

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.

        Args:
            indices: Experience indices
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority + 1e-6
            self._max_priority = max(self._max_priority, priority)


class TradingEnvironment:
    """Trading environment for RL agents."""

    def __init__(
        self,
        data: np.ndarray,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
        reward_type: RewardType = RewardType.RISK_ADJUSTED
    ):
        """
        Initialize trading environment.

        Args:
            data: Price/feature data (n_steps, n_features)
            initial_balance: Starting balance
            transaction_cost: Transaction cost percentage
            max_position: Maximum position size
            reward_type: Type of reward function
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_type = reward_type

        self._current_step = 0
        self._balance = initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._returns: list[float] = []

        self.n_steps = len(data)
        self.n_features = data.shape[1] if data.ndim > 1 else 1
        self.n_actions = 3

        logger.info(
            f"Initialized TradingEnvironment: steps={self.n_steps}, "
            f"features={self.n_features}"
        )

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial state
        """
        self._current_step = 0
        self._balance = self.initial_balance
        self._position = 0.0
        self._entry_price = 0.0
        self._total_trades = 0
        self._wins = 0
        self._losses = 0
        self._returns = []

        return self._get_state()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute action and return new state.

        Args:
            action: Action to execute (0=BUY, 1=SELL, 2=HOLD)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        current_price = self._get_current_price()

        old_portfolio_value = self._calculate_portfolio_value(current_price)

        trade_pnl = 0.0
        transaction_costs = 0.0

        if action == ActionType.BUY.value and self._position <= 0:
            if self._position < 0:
                trade_pnl = (self._entry_price - current_price) * abs(self._position)
                transaction_costs += abs(self._position) * current_price * self.transaction_cost
                self._position = 0

            shares_to_buy = (self._balance * self.max_position) / current_price
            transaction_costs += shares_to_buy * current_price * self.transaction_cost

            self._position = shares_to_buy
            self._entry_price = current_price
            self._balance -= shares_to_buy * current_price + transaction_costs
            self._total_trades += 1

        elif action == ActionType.SELL.value and self._position >= 0:
            if self._position > 0:
                trade_pnl = (current_price - self._entry_price) * self._position
                transaction_costs += self._position * current_price * self.transaction_cost

                self._balance += self._position * current_price - transaction_costs
                self._position = 0

                if trade_pnl > 0:
                    self._wins += 1
                else:
                    self._losses += 1

            shares_to_short = (self._balance * self.max_position) / current_price
            transaction_costs += shares_to_short * current_price * self.transaction_cost

            self._position = -shares_to_short
            self._entry_price = current_price
            self._balance -= transaction_costs
            self._total_trades += 1

        self._current_step += 1

        new_price = self._get_current_price()
        new_portfolio_value = self._calculate_portfolio_value(new_price)

        step_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value
        self._returns.append(step_return)

        reward = self._calculate_reward(step_return, trade_pnl, transaction_costs)

        done = self._current_step >= self.n_steps - 1

        if done and self._position != 0:
            final_pnl = (new_price - self._entry_price) * self._position
            if self._position > 0:
                self._balance += self._position * new_price
            else:
                self._balance += abs(self._position) * (2 * self._entry_price - new_price)
            self._position = 0

            if final_pnl > 0:
                self._wins += 1
            elif final_pnl < 0:
                self._losses += 1

        next_state = self._get_state()

        info = {
            "portfolio_value": new_portfolio_value,
            "balance": self._balance,
            "position": self._position,
            "trade_pnl": trade_pnl,
            "transaction_costs": transaction_costs,
            "total_trades": self._total_trades
        }

        return next_state, reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state."""
        if self._current_step >= self.n_steps:
            features = self.data[-1]
        else:
            features = self.data[self._current_step]

        if features.ndim == 0:
            features = np.array([features])

        position_normalized = self._position / self.max_position
        balance_normalized = self._balance / self.initial_balance - 1

        current_price = self._get_current_price()
        if self._position != 0:
            unrealized_pnl = (current_price - self._entry_price) * self._position / self.initial_balance
        else:
            unrealized_pnl = 0.0

        return np.concatenate([
            features.flatten(),
            np.array([position_normalized, balance_normalized, unrealized_pnl])
        ])

    def _get_current_price(self) -> float:
        """Get current price."""
        if self._current_step >= self.n_steps:
            return float(self.data[-1, 0]) if self.data.ndim > 1 else float(self.data[-1])
        return float(self.data[self._current_step, 0]) if self.data.ndim > 1 else float(self.data[self._current_step])

    def _calculate_portfolio_value(self, price: float) -> float:
        """Calculate total portfolio value."""
        position_value = self._position * price
        return self._balance + position_value

    def _calculate_reward(
        self,
        step_return: float,
        trade_pnl: float,
        transaction_costs: float
    ) -> float:
        """Calculate reward based on reward type."""
        if self.reward_type == RewardType.PROFIT:
            return step_return * 100

        elif self.reward_type == RewardType.SHARPE:
            if len(self._returns) < 2:
                return step_return * 100

            returns_array = np.array(self._returns)
            if np.std(returns_array) > 0:
                sharpe = np.mean(returns_array) / np.std(returns_array)
                return sharpe
            return step_return * 100

        elif self.reward_type == RewardType.SORTINO:
            if len(self._returns) < 2:
                return step_return * 100

            returns_array = np.array(self._returns)
            downside = returns_array[returns_array < 0]

            if len(downside) > 0 and np.std(downside) > 0:
                sortino = np.mean(returns_array) / np.std(downside)
                return sortino
            return step_return * 100

        elif self.reward_type == RewardType.RISK_ADJUSTED:
            reward = step_return * 100

            if step_return < 0:
                reward *= 1.5

            reward -= transaction_costs / self.initial_balance * 100

            return reward

        else:
            return step_return * 100

    def get_episode_result(self) -> EpisodeResult:
        """Get episode result."""
        returns_array = np.array(self._returns) if self._returns else np.array([0.0])

        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe = 0.0

        return EpisodeResult(
            total_reward=sum(self._returns),
            total_steps=self._current_step,
            final_balance=self._balance,
            trades=self._total_trades,
            wins=self._wins,
            losses=self._losses,
            sharpe_ratio=sharpe
        )


class BaseRLAgent(ABC):
    """Base class for RL agents."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        name: str = "RLAgent"
    ):
        """
        Initialize RL agent.

        Args:
            state_size: State dimension
            action_size: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            name: Agent name
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.name = name

        self._training_history: list[float] = []
        self._is_trained = False

        logger.info(
            f"Initialized {self.__class__.__name__}: {name}, "
            f"state_size={state_size}, action_size={action_size}"
        )

    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action given state."""
        pass

    @abstractmethod
    def update(self, experience: Experience) -> float:
        """Update agent from experience."""
        pass

    @abstractmethod
    async def train(
        self,
        env: TradingEnvironment,
        n_episodes: int
    ) -> list[EpisodeResult]:
        """Train agent."""
        pass

    @property
    def is_trained(self) -> bool:
        """Check if agent is trained."""
        return self._is_trained


class QLearningAgent(BaseRLAgent):
    """Q-Learning agent with tabular Q-function."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        n_bins: int = 10,
        name: str = "QLearning"
    ):
        """
        Initialize Q-Learning agent.

        Args:
            state_size: State dimension
            action_size: Number of actions
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            n_bins: Number of bins for state discretization
            name: Agent name
        """
        super().__init__(state_size, action_size, learning_rate, gamma, name)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_bins = n_bins

        q_table_shape = tuple([n_bins] * state_size + [action_size])
        self._q_table = np.zeros(q_table_shape)
        self._state_bins: list[np.ndarray] = []

    def _discretize_state(self, state: np.ndarray) -> tuple[int, ...]:
        """Discretize continuous state."""
        if len(self._state_bins) == 0:
            return tuple(np.clip(
                ((state + 1) / 2 * self.n_bins).astype(int),
                0, self.n_bins - 1
            ))

        discrete_state = []
        for i, val in enumerate(state):
            if i < len(self._state_bins):
                bin_idx = np.digitize(val, self._state_bins[i]) - 1
                bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)
            else:
                bin_idx = int(np.clip((val + 1) / 2 * self.n_bins, 0, self.n_bins - 1))
            discrete_state.append(bin_idx)

        return tuple(discrete_state)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        discrete_state = self._discretize_state(state)
        return int(np.argmax(self._q_table[discrete_state]))

    def update(self, experience: Experience) -> float:
        """
        Update Q-table from experience.

        Args:
            experience: Experience tuple

        Returns:
            TD error
        """
        state = self._discretize_state(experience.state)
        next_state = self._discretize_state(experience.next_state)
        action = experience.action

        current_q = self._q_table[state + (action,)]

        if experience.done:
            target = experience.reward
        else:
            max_next_q = np.max(self._q_table[next_state])
            target = experience.reward + self.gamma * max_next_q

        td_error = target - current_q

        self._q_table[state + (action,)] += self.learning_rate * td_error

        return float(abs(td_error))

    async def train(
        self,
        env: TradingEnvironment,
        n_episodes: int
    ) -> list[EpisodeResult]:
        """
        Train Q-Learning agent.

        Args:
            env: Trading environment
            n_episodes: Number of episodes

        Returns:
            List of episode results
        """
        logger.info(f"Training {self.name} for {n_episodes} episodes")

        results = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info
                )

                self.update(experience)

                total_reward += reward
                state = next_state

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_result = env.get_episode_result()
            results.append(episode_result)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean([r.total_reward for r in results[-100:]])
                logger.info(
                    f"Episode {episode + 1}: avg_reward={avg_reward:.2f}, "
                    f"epsilon={self.epsilon:.3f}"
                )

        self._is_trained = True
        logger.info(f"Training complete: {n_episodes} episodes")

        return results


class DQNAgent(BaseRLAgent):
    """Deep Q-Network agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        name: str = "DQN"
    ):
        """
        Initialize DQN agent.

        Args:
            state_size: State dimension
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            target_update_freq: Target network update frequency
            name: Agent name
        """
        super().__init__(state_size, action_size, learning_rate, gamma, name)

        self.hidden_sizes = hidden_sizes
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self._replay_buffer = ReplayBuffer(buffer_size)

        self._q_network = self._build_network()
        self._target_network = self._build_network()
        self._sync_networks()

        self._update_count = 0

    def _build_network(self) -> dict[str, np.ndarray]:
        """Build Q-network weights."""
        weights = {}

        layer_sizes = [self.state_size] + self.hidden_sizes + [self.action_size]

        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            weights[f"W{i}"] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            weights[f"b{i}"] = np.zeros(layer_sizes[i + 1])

        return weights

    def _sync_networks(self) -> None:
        """Sync target network with Q-network."""
        for key in self._q_network:
            self._target_network[key] = self._q_network[key].copy()

    def _forward(
        self,
        state: np.ndarray,
        network: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Forward pass through network."""
        x = state

        n_layers = len([k for k in network if k.startswith("W")])

        for i in range(n_layers - 1):
            x = x @ network[f"W{i}"] + network[f"b{i}"]
            x = np.maximum(0, x)

        x = x @ network[f"W{n_layers - 1}"] + network[f"b{n_layers - 1}"]

        return x

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        if state.ndim == 1:
            state = state.reshape(1, -1)

        q_values = self._forward(state, self._q_network)
        return int(np.argmax(q_values[0]))

    def update(self, experience: Experience) -> float:
        """
        Store experience and update network.

        Args:
            experience: Experience tuple

        Returns:
            Loss value
        """
        self._replay_buffer.push(experience)

        if len(self._replay_buffer) < self.batch_size:
            return 0.0

        batch = self._replay_buffer.sample(self.batch_size)

        states = np.array([e.state for e in batch])
        actions = np.array([e.action for e in batch])
        rewards = np.array([e.reward for e in batch])
        next_states = np.array([e.next_state for e in batch])
        dones = np.array([e.done for e in batch])

        current_q = self._forward(states, self._q_network)
        next_q = self._forward(next_states, self._target_network)

        targets = current_q.copy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])

        loss = self._update_network(states, targets, actions)

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self._sync_networks()

        return loss

    def _update_network(
        self,
        states: np.ndarray,
        targets: np.ndarray,
        actions: np.ndarray
    ) -> float:
        """Update Q-network using gradient descent."""
        current_q = self._forward(states, self._q_network)

        td_errors = np.zeros_like(current_q)
        for i in range(len(actions)):
            td_errors[i, actions[i]] = targets[i, actions[i]] - current_q[i, actions[i]]

        loss = float(np.mean(td_errors ** 2))

        n_layers = len([k for k in self._q_network if k.startswith("W")])

        grad = 2 * td_errors / len(states)

        activations = [states]
        x = states
        for i in range(n_layers - 1):
            x = x @ self._q_network[f"W{i}"] + self._q_network[f"b{i}"]
            x = np.maximum(0, x)
            activations.append(x)

        for i in range(n_layers - 1, -1, -1):
            dW = activations[i].T @ grad
            db = np.sum(grad, axis=0)

            self._q_network[f"W{i}"] += self.learning_rate * dW
            self._q_network[f"b{i}"] += self.learning_rate * db

            if i > 0:
                grad = grad @ self._q_network[f"W{i}"].T
                grad = grad * (activations[i] > 0)

        return loss

    async def train(
        self,
        env: TradingEnvironment,
        n_episodes: int
    ) -> list[EpisodeResult]:
        """
        Train DQN agent.

        Args:
            env: Trading environment
            n_episodes: Number of episodes

        Returns:
            List of episode results
        """
        logger.info(f"Training {self.name} for {n_episodes} episodes")

        results = []

        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0.0
            done = False
            total_loss = 0.0
            steps = 0

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info
                )

                loss = self.update(experience)
                total_loss += loss

                total_reward += reward
                state = next_state
                steps += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            episode_result = env.get_episode_result()
            results.append(episode_result)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean([r.total_reward for r in results[-50:]])
                avg_loss = total_loss / max(steps, 1)
                logger.info(
                    f"Episode {episode + 1}: avg_reward={avg_reward:.2f}, "
                    f"loss={avg_loss:.4f}, epsilon={self.epsilon:.3f}"
                )

        self._is_trained = True
        logger.info(f"Training complete: {n_episodes} episodes")

        return results


class PolicyGradientAgent(BaseRLAgent):
    """Policy Gradient (REINFORCE) agent."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: list[int] = [64, 64],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        name: str = "PolicyGradient"
    ):
        """
        Initialize Policy Gradient agent.

        Args:
            state_size: State dimension
            action_size: Number of actions
            hidden_sizes: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            name: Agent name
        """
        super().__init__(state_size, action_size, learning_rate, gamma, name)

        self.hidden_sizes = hidden_sizes
        self._policy_network = self._build_network()

        self._episode_states: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []

    def _build_network(self) -> dict[str, np.ndarray]:
        """Build policy network weights."""
        weights = {}
        layer_sizes = [self.state_size] + self.hidden_sizes + [self.action_size]

        for i in range(len(layer_sizes) - 1):
            scale = np.sqrt(2.0 / layer_sizes[i])
            weights[f"W{i}"] = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            weights[f"b{i}"] = np.zeros(layer_sizes[i + 1])

        return weights

    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass returning action probabilities."""
        x = state
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_layers = len([k for k in self._policy_network if k.startswith("W")])

        for i in range(n_layers - 1):
            x = x @ self._policy_network[f"W{i}"] + self._policy_network[f"b{i}"]
            x = np.maximum(0, x)

        x = x @ self._policy_network[f"W{n_layers - 1}"] + self._policy_network[f"b{n_layers - 1}"]

        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        return probs

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action by sampling from policy.

        Args:
            state: Current state
            training: Whether in training mode

        Returns:
            Selected action
        """
        probs = self._forward(state)[0]

        if training:
            action = np.random.choice(self.action_size, p=probs)
        else:
            action = np.argmax(probs)

        return int(action)

    def update(self, experience: Experience) -> float:
        """
        Store experience for episode.

        Args:
            experience: Experience tuple

        Returns:
            0.0 (update happens at episode end)
        """
        self._episode_states.append(experience.state)
        self._episode_actions.append(experience.action)
        self._episode_rewards.append(experience.reward)

        return 0.0

    def _update_policy(self) -> float:
        """Update policy network at episode end."""
        if len(self._episode_rewards) == 0:
            return 0.0

        returns = []
        G = 0
        for r in reversed(self._episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)

        total_loss = 0.0

        for state, action, G in zip(
            self._episode_states,
            self._episode_actions,
            returns
        ):
            state = state.reshape(1, -1)
            probs = self._forward(state)[0]

            grad_log_prob = -probs.copy()
            grad_log_prob[action] += 1

            grad = grad_log_prob * G

            self._update_network_gradient(state, grad)

            total_loss += -np.log(probs[action] + 1e-8) * G

        self._episode_states = []
        self._episode_actions = []
        self._episode_rewards = []

        return float(total_loss / len(returns)) if len(returns) > 0 else 0.0

    def _update_network_gradient(
        self,
        state: np.ndarray,
        output_grad: np.ndarray
    ) -> None:
        """Update network with gradient."""
        n_layers = len([k for k in self._policy_network if k.startswith("W")])

        activations = [state]
        x = state
        for i in range(n_layers - 1):
            x = x @ self._policy_network[f"W{i}"] + self._policy_network[f"b{i}"]
            x = np.maximum(0, x)
            activations.append(x)

        grad = output_grad.reshape(1, -1)

        for i in range(n_layers - 1, -1, -1):
            dW = activations[i].T @ grad
            db = grad.flatten()

            self._policy_network[f"W{i}"] += self.learning_rate * dW
            self._policy_network[f"b{i}"] += self.learning_rate * db

            if i > 0:
                grad = grad @ self._policy_network[f"W{i}"].T
                grad = grad * (activations[i] > 0)

    async def train(
        self,
        env: TradingEnvironment,
        n_episodes: int
    ) -> list[EpisodeResult]:
        """
        Train Policy Gradient agent.

        Args:
            env: Trading environment
            n_episodes: Number of episodes

        Returns:
            List of episode results
        """
        logger.info(f"Training {self.name} for {n_episodes} episodes")

        results = []

        for episode in range(n_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)

                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=info
                )

                self.update(experience)
                state = next_state

            loss = self._update_policy()

            episode_result = env.get_episode_result()
            results.append(episode_result)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean([r.total_reward for r in results[-50:]])
                logger.info(
                    f"Episode {episode + 1}: avg_reward={avg_reward:.2f}, "
                    f"loss={loss:.4f}"
                )

        self._is_trained = True
        logger.info(f"Training complete: {n_episodes} episodes")

        return results


def create_trading_environment(
    data: np.ndarray,
    initial_balance: float = 100000.0,
    reward_type: str = "risk_adjusted"
) -> TradingEnvironment:
    """
    Factory function to create trading environment.

    Args:
        data: Price/feature data
        initial_balance: Starting balance
        reward_type: Type of reward function

    Returns:
        TradingEnvironment instance
    """
    return TradingEnvironment(
        data=data,
        initial_balance=initial_balance,
        reward_type=RewardType(reward_type)
    )


def create_dqn_agent(
    state_size: int,
    action_size: int = 3,
    hidden_sizes: list[int] = [64, 64],
    learning_rate: float = 0.001,
    name: str = "DQN"
) -> DQNAgent:
    """
    Factory function to create DQN agent.

    Args:
        state_size: State dimension
        action_size: Number of actions
        hidden_sizes: Hidden layer sizes
        learning_rate: Learning rate
        name: Agent name

    Returns:
        DQNAgent instance
    """
    return DQNAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        name=name
    )


def create_policy_gradient_agent(
    state_size: int,
    action_size: int = 3,
    hidden_sizes: list[int] = [64, 64],
    learning_rate: float = 0.001,
    name: str = "PolicyGradient"
) -> PolicyGradientAgent:
    """
    Factory function to create Policy Gradient agent.

    Args:
        state_size: State dimension
        action_size: Number of actions
        hidden_sizes: Hidden layer sizes
        learning_rate: Learning rate
        name: Agent name

    Returns:
        PolicyGradientAgent instance
    """
    return PolicyGradientAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        name=name
    )
