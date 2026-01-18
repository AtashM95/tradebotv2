"""
Deep Learning Module for Ultimate Trading Bot v2.2

Implements LSTM, GRU, CNN, Transformer, and hybrid neural network models
for time series prediction and trading signal generation.

Author: AI Assistant
Version: 2.2.0
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class ActivationType(Enum):
    """Types of activation functions."""
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SOFTMAX = "softmax"
    LINEAR = "linear"
    GELU = "gelu"


class LayerType(Enum):
    """Types of neural network layers."""
    DENSE = "dense"
    LSTM = "lstm"
    GRU = "gru"
    CONV1D = "conv1d"
    ATTENTION = "attention"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    EMBEDDING = "embedding"


class OptimizerType(Enum):
    """Types of optimizers."""
    SGD = "sgd"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADAMW = "adamw"


class LossType(Enum):
    """Types of loss functions."""
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    layer_type: LayerType
    units: int
    activation: ActivationType = ActivationType.RELU
    dropout_rate: float = 0.0
    return_sequences: bool = False
    kernel_size: int = 3
    filters: int = 32
    padding: str = "same"
    use_bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_type": self.layer_type.value,
            "units": self.units,
            "activation": self.activation.value,
            "dropout_rate": self.dropout_rate,
            "return_sequences": self.return_sequences,
            "kernel_size": self.kernel_size,
            "filters": self.filters,
            "padding": self.padding,
            "use_bias": self.use_bias
        }


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    loss: LossType = LossType.MSE
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    weight_decay: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer.value,
            "loss": self.loss.value,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "min_lr": self.min_lr,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay
        }


@dataclass
class TrainingHistory:
    """Training history."""
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    training_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loss": self.loss,
            "val_loss": self.val_loss,
            "metrics": self.metrics,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "training_time": self.training_time
        }


@dataclass
class DeepLearningPrediction:
    """Deep learning prediction result."""
    predictions: np.ndarray
    confidence: np.ndarray
    model_name: str
    prediction_time: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "predictions": self.predictions.tolist(),
            "confidence": self.confidence.tolist(),
            "model_name": self.model_name,
            "prediction_time": self.prediction_time
        }


class ActivationFunctions:
    """Collection of activation functions."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation."""
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(float)

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU activation."""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU derivative."""
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Sigmoid derivative."""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Tanh derivative."""
        return 1 - np.tanh(x) ** 2

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation."""
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """ELU activation."""
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def get_activation(activation_type: ActivationType) -> Callable:
        """Get activation function by type."""
        activations = {
            ActivationType.RELU: ActivationFunctions.relu,
            ActivationType.TANH: ActivationFunctions.tanh,
            ActivationType.SIGMOID: ActivationFunctions.sigmoid,
            ActivationType.LEAKY_RELU: ActivationFunctions.leaky_relu,
            ActivationType.ELU: ActivationFunctions.elu,
            ActivationType.SOFTMAX: ActivationFunctions.softmax,
            ActivationType.LINEAR: lambda x: x,
            ActivationType.GELU: ActivationFunctions.gelu
        }
        return activations.get(activation_type, ActivationFunctions.relu)


class LossFunctions:
    """Collection of loss functions."""

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean squared error."""
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mse_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """MSE gradient."""
        return 2 * (y_pred - y_true) / y_true.size

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean absolute error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mae_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """MAE gradient."""
        return np.sign(y_pred - y_true) / y_true.size

    @staticmethod
    def huber(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        delta: float = 1.0
    ) -> float:
        """Huber loss."""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return float(np.mean(np.where(is_small_error, squared_loss, linear_loss)))

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """Binary cross entropy."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cross entropy loss."""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=-1)))

    @staticmethod
    def get_loss(loss_type: LossType) -> Callable:
        """Get loss function by type."""
        losses = {
            LossType.MSE: LossFunctions.mse,
            LossType.MAE: LossFunctions.mae,
            LossType.HUBER: LossFunctions.huber,
            LossType.CROSS_ENTROPY: LossFunctions.cross_entropy,
            LossType.BINARY_CROSS_ENTROPY: LossFunctions.binary_cross_entropy
        }
        return losses.get(loss_type, LossFunctions.mse)


class BaseLayer(ABC):
    """Base class for neural network layers."""

    def __init__(self, name: str = "layer"):
        """
        Initialize layer.

        Args:
            name: Layer name
        """
        self.name = name
        self._trainable = True
        self._weights: dict[str, np.ndarray] = {}
        self._gradients: dict[str, np.ndarray] = {}
        self._cache: dict[str, Any] = {}

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass."""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        pass

    @property
    def weights(self) -> dict[str, np.ndarray]:
        """Get layer weights."""
        return self._weights

    @property
    def gradients(self) -> dict[str, np.ndarray]:
        """Get layer gradients."""
        return self._gradients


class DenseLayer(BaseLayer):
    """Fully connected layer."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationType = ActivationType.RELU,
        use_bias: bool = True,
        name: str = "dense"
    ):
        """
        Initialize dense layer.

        Args:
            input_size: Input dimension
            output_size: Output dimension
            activation: Activation function
            use_bias: Whether to use bias
            name: Layer name
        """
        super().__init__(name)

        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation
        self.use_bias = use_bias

        self.activation = ActivationFunctions.get_activation(activation)

        scale = np.sqrt(2.0 / input_size)
        self._weights["W"] = np.random.randn(input_size, output_size) * scale

        if use_bias:
            self._weights["b"] = np.zeros(output_size)

        logger.debug(f"Initialized DenseLayer: {input_size} -> {output_size}")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        self._cache["input"] = x

        z = x @ self._weights["W"]
        if self.use_bias:
            z = z + self._weights["b"]

        self._cache["pre_activation"] = z

        output = self.activation(z)
        self._cache["output"] = output

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self.activation_type == ActivationType.RELU:
            activation_grad = ActivationFunctions.relu_derivative(
                self._cache["pre_activation"]
            )
        elif self.activation_type == ActivationType.SIGMOID:
            activation_grad = ActivationFunctions.sigmoid_derivative(
                self._cache["pre_activation"]
            )
        elif self.activation_type == ActivationType.TANH:
            activation_grad = ActivationFunctions.tanh_derivative(
                self._cache["pre_activation"]
            )
        else:
            activation_grad = np.ones_like(self._cache["pre_activation"])

        delta = grad * activation_grad

        input_data = self._cache["input"]
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
        if delta.ndim == 1:
            delta = delta.reshape(1, -1)

        self._gradients["W"] = input_data.T @ delta

        if self.use_bias:
            self._gradients["b"] = np.sum(delta, axis=0)

        input_grad = delta @ self._weights["W"].T

        return input_grad


class LSTMLayer(BaseLayer):
    """Long Short-Term Memory layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        return_sequences: bool = False,
        name: str = "lstm"
    ):
        """
        Initialize LSTM layer.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            return_sequences: Whether to return full sequence
            name: Layer name
        """
        super().__init__(name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        scale = np.sqrt(2.0 / (input_size + hidden_size))

        self._weights["Wf"] = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self._weights["Wi"] = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self._weights["Wc"] = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self._weights["Wo"] = np.random.randn(input_size + hidden_size, hidden_size) * scale

        self._weights["bf"] = np.ones(hidden_size)
        self._weights["bi"] = np.zeros(hidden_size)
        self._weights["bc"] = np.zeros(hidden_size)
        self._weights["bo"] = np.zeros(hidden_size)

        logger.debug(f"Initialized LSTMLayer: {input_size} -> {hidden_size}")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])

        batch_size, seq_len, _ = x.shape

        h = np.zeros((batch_size, self.hidden_size))
        c = np.zeros((batch_size, self.hidden_size))

        self._cache["inputs"] = []
        self._cache["gates"] = []
        self._cache["states"] = [(h.copy(), c.copy())]

        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]

            concat = np.concatenate([xt, h], axis=1)
            self._cache["inputs"].append(concat)

            ft = ActivationFunctions.sigmoid(concat @ self._weights["Wf"] + self._weights["bf"])
            it = ActivationFunctions.sigmoid(concat @ self._weights["Wi"] + self._weights["bi"])
            ct_tilde = ActivationFunctions.tanh(concat @ self._weights["Wc"] + self._weights["bc"])
            ot = ActivationFunctions.sigmoid(concat @ self._weights["Wo"] + self._weights["bo"])

            c = ft * c + it * ct_tilde
            h = ot * ActivationFunctions.tanh(c)

            self._cache["gates"].append((ft, it, ct_tilde, ot))
            self._cache["states"].append((h.copy(), c.copy()))

            outputs.append(h)

        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self.return_sequences:
            seq_len = grad.shape[1]
            batch_size = grad.shape[0]
        else:
            seq_len = len(self._cache["gates"])
            batch_size = grad.shape[0]
            grad_seq = np.zeros((batch_size, seq_len, self.hidden_size))
            grad_seq[:, -1, :] = grad
            grad = grad_seq

        for key in ["Wf", "Wi", "Wc", "Wo", "bf", "bi", "bc", "bo"]:
            self._gradients[key] = np.zeros_like(self._weights[key])

        dh_next = np.zeros((batch_size, self.hidden_size))
        dc_next = np.zeros((batch_size, self.hidden_size))

        input_grads = []

        for t in reversed(range(seq_len)):
            dh = grad[:, t, :] + dh_next

            h_t, c_t = self._cache["states"][t + 1]
            _, c_prev = self._cache["states"][t]
            ft, it, ct_tilde, ot = self._cache["gates"][t]
            concat = self._cache["inputs"][t]

            do = dh * ActivationFunctions.tanh(c_t)
            do = do * ot * (1 - ot)

            dc = dh * ot * (1 - ActivationFunctions.tanh(c_t) ** 2) + dc_next

            df = dc * c_prev
            df = df * ft * (1 - ft)

            di = dc * ct_tilde
            di = di * it * (1 - it)

            dc_tilde = dc * it
            dc_tilde = dc_tilde * (1 - ct_tilde ** 2)

            self._gradients["Wf"] += concat.T @ df
            self._gradients["Wi"] += concat.T @ di
            self._gradients["Wc"] += concat.T @ dc_tilde
            self._gradients["Wo"] += concat.T @ do

            self._gradients["bf"] += np.sum(df, axis=0)
            self._gradients["bi"] += np.sum(di, axis=0)
            self._gradients["bc"] += np.sum(dc_tilde, axis=0)
            self._gradients["bo"] += np.sum(do, axis=0)

            d_concat = (df @ self._weights["Wf"].T +
                       di @ self._weights["Wi"].T +
                       dc_tilde @ self._weights["Wc"].T +
                       do @ self._weights["Wo"].T)

            dx = d_concat[:, :self.input_size]
            dh_next = d_concat[:, self.input_size:]
            dc_next = dc * ft

            input_grads.append(dx)

        input_grads = list(reversed(input_grads))
        return np.stack(input_grads, axis=1)


class GRULayer(BaseLayer):
    """Gated Recurrent Unit layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        return_sequences: bool = False,
        name: str = "gru"
    ):
        """
        Initialize GRU layer.

        Args:
            input_size: Input dimension
            hidden_size: Hidden state dimension
            return_sequences: Whether to return full sequence
            name: Layer name
        """
        super().__init__(name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences

        scale = np.sqrt(2.0 / (input_size + hidden_size))

        self._weights["Wz"] = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self._weights["Wr"] = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self._weights["Wh"] = np.random.randn(input_size + hidden_size, hidden_size) * scale

        self._weights["bz"] = np.zeros(hidden_size)
        self._weights["br"] = np.zeros(hidden_size)
        self._weights["bh"] = np.zeros(hidden_size)

        logger.debug(f"Initialized GRULayer: {input_size} -> {hidden_size}")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 1, x.shape[1])

        batch_size, seq_len, _ = x.shape

        h = np.zeros((batch_size, self.hidden_size))

        self._cache["inputs"] = []
        self._cache["gates"] = []
        self._cache["states"] = [h.copy()]

        outputs = []

        for t in range(seq_len):
            xt = x[:, t, :]

            concat = np.concatenate([xt, h], axis=1)
            self._cache["inputs"].append(concat)

            zt = ActivationFunctions.sigmoid(concat @ self._weights["Wz"] + self._weights["bz"])
            rt = ActivationFunctions.sigmoid(concat @ self._weights["Wr"] + self._weights["br"])

            concat_reset = np.concatenate([xt, rt * h], axis=1)
            ht_tilde = ActivationFunctions.tanh(concat_reset @ self._weights["Wh"] + self._weights["bh"])

            h = (1 - zt) * h + zt * ht_tilde

            self._cache["gates"].append((zt, rt, ht_tilde, concat_reset))
            self._cache["states"].append(h.copy())

            outputs.append(h)

        if self.return_sequences:
            return np.stack(outputs, axis=1)
        else:
            return outputs[-1]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self.return_sequences:
            seq_len = grad.shape[1]
            batch_size = grad.shape[0]
        else:
            seq_len = len(self._cache["gates"])
            batch_size = grad.shape[0]
            grad_seq = np.zeros((batch_size, seq_len, self.hidden_size))
            grad_seq[:, -1, :] = grad
            grad = grad_seq

        for key in ["Wz", "Wr", "Wh", "bz", "br", "bh"]:
            self._gradients[key] = np.zeros_like(self._weights[key])

        dh_next = np.zeros((batch_size, self.hidden_size))
        input_grads = []

        for t in reversed(range(seq_len)):
            dh = grad[:, t, :] + dh_next

            h_prev = self._cache["states"][t]
            zt, rt, ht_tilde, concat_reset = self._cache["gates"][t]
            concat = self._cache["inputs"][t]

            dz = dh * (ht_tilde - h_prev)
            dz = dz * zt * (1 - zt)

            dh_tilde = dh * zt
            dh_tilde = dh_tilde * (1 - ht_tilde ** 2)

            self._gradients["Wz"] += concat.T @ dz
            self._gradients["Wh"] += concat_reset.T @ dh_tilde

            self._gradients["bz"] += np.sum(dz, axis=0)
            self._gradients["bh"] += np.sum(dh_tilde, axis=0)

            d_concat_reset = dh_tilde @ self._weights["Wh"].T
            dx_from_h = d_concat_reset[:, :self.input_size]
            dh_reset = d_concat_reset[:, self.input_size:]

            dr = dh_reset * h_prev
            dr = dr * rt * (1 - rt)

            self._gradients["Wr"] += concat.T @ dr
            self._gradients["br"] += np.sum(dr, axis=0)

            d_concat = dz @ self._weights["Wz"].T + dr @ self._weights["Wr"].T
            dx = d_concat[:, :self.input_size] + dx_from_h
            dh_from_concat = d_concat[:, self.input_size:]

            dh_next = dh * (1 - zt) + dh_reset * rt + dh_from_concat

            input_grads.append(dx)

        input_grads = list(reversed(input_grads))
        return np.stack(input_grads, axis=1)


class DropoutLayer(BaseLayer):
    """Dropout layer for regularization."""

    def __init__(self, rate: float = 0.5, name: str = "dropout"):
        """
        Initialize dropout layer.

        Args:
            rate: Dropout rate
            name: Layer name
        """
        super().__init__(name)
        self.rate = rate
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        if training and self.rate > 0:
            self._mask = np.random.binomial(1, 1 - self.rate, x.shape) / (1 - self.rate)
            return x * self._mask
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        if self._mask is not None:
            return grad * self._mask
        return grad


class BatchNormLayer(BaseLayer):
    """Batch normalization layer."""

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-5,
        momentum: float = 0.1,
        name: str = "batch_norm"
    ):
        """
        Initialize batch norm layer.

        Args:
            num_features: Number of features
            epsilon: Small constant for numerical stability
            momentum: Momentum for running statistics
            name: Layer name
        """
        super().__init__(name)

        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum

        self._weights["gamma"] = np.ones(num_features)
        self._weights["beta"] = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

            self._cache["mean"] = mean
            self._cache["var"] = var
        else:
            mean = self.running_mean
            var = self.running_var

        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        self._cache["x_norm"] = x_norm
        self._cache["input"] = x

        return self._weights["gamma"] * x_norm + self._weights["beta"]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        x_norm = self._cache["x_norm"]
        x = self._cache["input"]
        mean = self._cache["mean"]
        var = self._cache["var"]

        batch_size = x.shape[0]

        self._gradients["gamma"] = np.sum(grad * x_norm, axis=0)
        self._gradients["beta"] = np.sum(grad, axis=0)

        dx_norm = grad * self._weights["gamma"]

        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std_inv ** 3, axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)

        dx = dx_norm * std_inv + dvar * 2 * (x - mean) / batch_size + dmean / batch_size

        return dx


class MultiHeadAttention(BaseLayer):
    """Multi-head attention layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        name: str = "attention"
    ):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            name: Layer name
        """
        super().__init__(name)

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(2.0 / d_model)

        self._weights["Wq"] = np.random.randn(d_model, d_model) * scale
        self._weights["Wk"] = np.random.randn(d_model, d_model) * scale
        self._weights["Wv"] = np.random.randn(d_model, d_model) * scale
        self._weights["Wo"] = np.random.randn(d_model, d_model) * scale

        logger.debug(f"Initialized MultiHeadAttention: d_model={d_model}, heads={num_heads}")

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            training: Whether in training mode

        Returns:
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        Q = x @ self._weights["Wq"]
        K = x @ self._weights["Wk"]
        V = x @ self._weights["Wv"]

        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        attention_weights = ActivationFunctions.softmax(scores)

        context = attention_weights @ V

        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        output = context @ self._weights["Wo"]

        self._cache["input"] = x
        self._cache["Q"] = Q
        self._cache["K"] = K
        self._cache["V"] = V
        self._cache["attention_weights"] = attention_weights
        self._cache["context"] = context

        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass.

        Args:
            grad: Gradient from next layer

        Returns:
            Gradient for previous layer
        """
        batch_size, seq_len, _ = grad.shape

        x = self._cache["input"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        attention_weights = self._cache["attention_weights"]
        context = self._cache["context"]

        self._gradients["Wo"] = context.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_model)

        d_context = grad @ self._weights["Wo"].T
        d_context = d_context.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        dV = attention_weights.transpose(0, 1, 3, 2) @ d_context
        d_attention = d_context @ V.transpose(0, 1, 3, 2)

        d_scores = d_attention * attention_weights * (1 - attention_weights)

        dQ = d_scores @ K / np.sqrt(self.d_k)
        dK = d_scores.transpose(0, 1, 3, 2) @ Q / np.sqrt(self.d_k)

        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        self._gradients["Wq"] = x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        self._gradients["Wk"] = x.reshape(-1, self.d_model).T @ dK.reshape(-1, self.d_model)
        self._gradients["Wv"] = x.reshape(-1, self.d_model).T @ dV.reshape(-1, self.d_model)

        dx = (dQ @ self._weights["Wq"].T +
              dK @ self._weights["Wk"].T +
              dV @ self._weights["Wv"].T)

        return dx


class BaseOptimizer(ABC):
    """Base class for optimizers."""

    def __init__(self, learning_rate: float = 0.001):
        """
        Initialize optimizer.

        Args:
            learning_rate: Learning rate
        """
        self.learning_rate = learning_rate

    @abstractmethod
    def update(
        self,
        weights: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Update weights."""
        pass


class SGDOptimizer(BaseOptimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """
        Initialize SGD.

        Args:
            learning_rate: Learning rate
            momentum: Momentum factor
            weight_decay: L2 regularization
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[str, np.ndarray] = {}

    def update(
        self,
        weights: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Update weights."""
        for key in weights:
            if key not in self._velocity:
                self._velocity[key] = np.zeros_like(weights[key])

            grad = gradients.get(key, np.zeros_like(weights[key]))

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * weights[key]

            self._velocity[key] = self.momentum * self._velocity[key] - self.learning_rate * grad
            weights[key] = weights[key] + self._velocity[key]

        return weights


class AdamOptimizer(BaseOptimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam.

        Args:
            learning_rate: Learning rate
            beta1: First moment decay
            beta2: Second moment decay
            epsilon: Small constant
            weight_decay: Weight decay (L2)
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t = 0

    def update(
        self,
        weights: dict[str, np.ndarray],
        gradients: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Update weights."""
        self._t += 1

        for key in weights:
            if key not in self._m:
                self._m[key] = np.zeros_like(weights[key])
                self._v[key] = np.zeros_like(weights[key])

            grad = gradients.get(key, np.zeros_like(weights[key]))

            if self.weight_decay > 0:
                grad = grad + self.weight_decay * weights[key]

            self._m[key] = self.beta1 * self._m[key] + (1 - self.beta1) * grad
            self._v[key] = self.beta2 * self._v[key] + (1 - self.beta2) * (grad ** 2)

            m_hat = self._m[key] / (1 - self.beta1 ** self._t)
            v_hat = self._v[key] / (1 - self.beta2 ** self._t)

            weights[key] = weights[key] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights


class BaseDeepLearningModel(ABC):
    """Base class for deep learning models."""

    def __init__(
        self,
        name: str,
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize model.

        Args:
            name: Model name
            training_config: Training configuration
        """
        self.name = name
        self.training_config = training_config or TrainingConfig()
        self._layers: list[BaseLayer] = []
        self._optimizer: Optional[BaseOptimizer] = None
        self._is_trained = False
        self._training_history: Optional[TrainingHistory] = None

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    def _build_model(self, input_shape: tuple[int, ...]) -> None:
        """Build the model architecture."""
        pass

    def _forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor
        """
        for layer in self._layers:
            x = layer.forward(x, training)
        return x

    def _backward(self, grad: np.ndarray) -> None:
        """
        Backward pass through all layers.

        Args:
            grad: Output gradient
        """
        for layer in reversed(self._layers):
            grad = layer.backward(grad)

    def _update_weights(self) -> None:
        """Update weights using optimizer."""
        if self._optimizer is None:
            return

        for layer in self._layers:
            if layer._trainable and len(layer.weights) > 0:
                layer._weights = self._optimizer.update(
                    layer.weights,
                    layer.gradients
                )

    async def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_data: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation data

        Returns:
            TrainingHistory object
        """
        try:
            import time
            start_time = time.time()

            logger.info(f"Training {self.name} on {len(X)} samples")

            if len(self._layers) == 0:
                input_shape = X.shape[1:] if X.ndim > 1 else (X.shape[0],)
                self._build_model(input_shape)

            if self._optimizer is None:
                if self.training_config.optimizer == OptimizerType.ADAM:
                    self._optimizer = AdamOptimizer(
                        learning_rate=self.training_config.learning_rate,
                        weight_decay=self.training_config.weight_decay
                    )
                else:
                    self._optimizer = SGDOptimizer(
                        learning_rate=self.training_config.learning_rate,
                        weight_decay=self.training_config.weight_decay
                    )

            loss_fn = LossFunctions.get_loss(self.training_config.loss)

            history = TrainingHistory()
            best_weights: list[dict[str, np.ndarray]] = []
            patience_counter = 0

            n_samples = len(X)
            batch_size = self.training_config.batch_size

            for epoch in range(self.training_config.epochs):
                indices = np.random.permutation(n_samples)
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i + batch_size]
                    X_batch = X[batch_indices]
                    y_batch = y[batch_indices]

                    y_pred = self._forward(X_batch, training=True)

                    loss = loss_fn(y_batch, y_pred)
                    epoch_loss += loss
                    n_batches += 1

                    if self.training_config.loss == LossType.MSE:
                        grad = LossFunctions.mse_gradient(y_batch, y_pred)
                    elif self.training_config.loss == LossType.MAE:
                        grad = LossFunctions.mae_gradient(y_batch, y_pred)
                    else:
                        grad = LossFunctions.mse_gradient(y_batch, y_pred)

                    grad = np.clip(grad, -self.training_config.gradient_clip,
                                  self.training_config.gradient_clip)

                    self._backward(grad)
                    self._update_weights()

                epoch_loss /= n_batches
                history.loss.append(epoch_loss)

                if validation_data is not None:
                    X_val, y_val = validation_data
                    y_val_pred = self._forward(X_val, training=False)
                    val_loss = loss_fn(y_val, y_val_pred)
                    history.val_loss.append(val_loss)

                    if val_loss < history.best_val_loss:
                        history.best_val_loss = val_loss
                        history.best_epoch = epoch
                        best_weights = [layer.weights.copy() for layer in self._layers]
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.training_config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                if (epoch + 1) % 10 == 0:
                    val_str = f", val_loss={history.val_loss[-1]:.4f}" if history.val_loss else ""
                    logger.info(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}{val_str}")

            if best_weights:
                for i, layer in enumerate(self._layers):
                    layer._weights = best_weights[i]

            history.training_time = time.time() - start_time
            self._training_history = history
            self._is_trained = True

            logger.info(
                f"Training complete: best_val_loss={history.best_val_loss:.4f}, "
                f"time={history.training_time:.2f}s"
            )

            return history

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    async def predict(
        self,
        X: np.ndarray
    ) -> DeepLearningPrediction:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            DeepLearningPrediction object
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            import time
            start_time = time.time()

            predictions = self._forward(X, training=False)

            confidence = np.ones(len(predictions)) * 0.5

            if predictions.ndim > 1 and predictions.shape[1] > 1:
                confidence = np.max(ActivationFunctions.softmax(predictions), axis=1)
            elif self.training_config.loss == LossType.BINARY_CROSS_ENTROPY:
                probs = ActivationFunctions.sigmoid(predictions.flatten())
                confidence = np.maximum(probs, 1 - probs)

            prediction_time = time.time() - start_time

            return DeepLearningPrediction(
                predictions=predictions,
                confidence=confidence,
                model_name=self.name,
                prediction_time=prediction_time
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained

    @property
    def training_history(self) -> Optional[TrainingHistory]:
        """Get training history."""
        return self._training_history


class LSTMModel(BaseDeepLearningModel):
    """LSTM model for sequence prediction."""

    def __init__(
        self,
        hidden_sizes: list[int],
        output_size: int = 1,
        dropout_rate: float = 0.2,
        name: str = "LSTM",
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize LSTM model.

        Args:
            hidden_sizes: List of hidden layer sizes
            output_size: Output dimension
            dropout_rate: Dropout rate
            name: Model name
            training_config: Training configuration
        """
        super().__init__(name, training_config)

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def _build_model(self, input_shape: tuple[int, ...]) -> None:
        """Build LSTM architecture."""
        if len(input_shape) == 2:
            input_size = input_shape[1]
        else:
            input_size = input_shape[0]

        current_size = input_size

        for i, hidden_size in enumerate(self.hidden_sizes):
            return_sequences = i < len(self.hidden_sizes) - 1

            self._layers.append(LSTMLayer(
                input_size=current_size,
                hidden_size=hidden_size,
                return_sequences=return_sequences,
                name=f"lstm_{i}"
            ))

            if self.dropout_rate > 0:
                self._layers.append(DropoutLayer(
                    rate=self.dropout_rate,
                    name=f"dropout_{i}"
                ))

            current_size = hidden_size

        self._layers.append(DenseLayer(
            input_size=current_size,
            output_size=self.output_size,
            activation=ActivationType.LINEAR,
            name="output"
        ))

        logger.info(f"Built LSTM model with {len(self._layers)} layers")


class GRUModel(BaseDeepLearningModel):
    """GRU model for sequence prediction."""

    def __init__(
        self,
        hidden_sizes: list[int],
        output_size: int = 1,
        dropout_rate: float = 0.2,
        name: str = "GRU",
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize GRU model.

        Args:
            hidden_sizes: List of hidden layer sizes
            output_size: Output dimension
            dropout_rate: Dropout rate
            name: Model name
            training_config: Training configuration
        """
        super().__init__(name, training_config)

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def _build_model(self, input_shape: tuple[int, ...]) -> None:
        """Build GRU architecture."""
        if len(input_shape) == 2:
            input_size = input_shape[1]
        else:
            input_size = input_shape[0]

        current_size = input_size

        for i, hidden_size in enumerate(self.hidden_sizes):
            return_sequences = i < len(self.hidden_sizes) - 1

            self._layers.append(GRULayer(
                input_size=current_size,
                hidden_size=hidden_size,
                return_sequences=return_sequences,
                name=f"gru_{i}"
            ))

            if self.dropout_rate > 0:
                self._layers.append(DropoutLayer(
                    rate=self.dropout_rate,
                    name=f"dropout_{i}"
                ))

            current_size = hidden_size

        self._layers.append(DenseLayer(
            input_size=current_size,
            output_size=self.output_size,
            activation=ActivationType.LINEAR,
            name="output"
        ))

        logger.info(f"Built GRU model with {len(self._layers)} layers")


class TransformerModel(BaseDeepLearningModel):
    """Transformer model for sequence prediction."""

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 128,
        output_size: int = 1,
        dropout_rate: float = 0.1,
        name: str = "Transformer",
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize Transformer model.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            output_size: Output dimension
            dropout_rate: Dropout rate
            name: Model name
            training_config: Training configuration
        """
        super().__init__(name, training_config)

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.output_size = output_size
        self.dropout_rate = dropout_rate

    def _build_model(self, input_shape: tuple[int, ...]) -> None:
        """Build Transformer architecture."""
        if len(input_shape) == 2:
            input_size = input_shape[1]
        else:
            input_size = input_shape[0]

        self._layers.append(DenseLayer(
            input_size=input_size,
            output_size=self.d_model,
            activation=ActivationType.LINEAR,
            name="input_projection"
        ))

        for i in range(self.num_layers):
            self._layers.append(MultiHeadAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                name=f"attention_{i}"
            ))

            if self.dropout_rate > 0:
                self._layers.append(DropoutLayer(
                    rate=self.dropout_rate,
                    name=f"attn_dropout_{i}"
                ))

            self._layers.append(DenseLayer(
                input_size=self.d_model,
                output_size=self.ff_dim,
                activation=ActivationType.GELU,
                name=f"ff1_{i}"
            ))

            self._layers.append(DenseLayer(
                input_size=self.ff_dim,
                output_size=self.d_model,
                activation=ActivationType.LINEAR,
                name=f"ff2_{i}"
            ))

            if self.dropout_rate > 0:
                self._layers.append(DropoutLayer(
                    rate=self.dropout_rate,
                    name=f"ff_dropout_{i}"
                ))

        self._layers.append(DenseLayer(
            input_size=self.d_model,
            output_size=self.output_size,
            activation=ActivationType.LINEAR,
            name="output"
        ))

        logger.info(f"Built Transformer model with {len(self._layers)} layers")


class MLPModel(BaseDeepLearningModel):
    """Multi-layer perceptron model."""

    def __init__(
        self,
        hidden_sizes: list[int],
        output_size: int = 1,
        activation: ActivationType = ActivationType.RELU,
        dropout_rate: float = 0.2,
        name: str = "MLP",
        training_config: Optional[TrainingConfig] = None
    ):
        """
        Initialize MLP model.

        Args:
            hidden_sizes: List of hidden layer sizes
            output_size: Output dimension
            activation: Activation function
            dropout_rate: Dropout rate
            name: Model name
            training_config: Training configuration
        """
        super().__init__(name, training_config)

        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout_rate = dropout_rate

    def _build_model(self, input_shape: tuple[int, ...]) -> None:
        """Build MLP architecture."""
        input_size = int(np.prod(input_shape))
        current_size = input_size

        for i, hidden_size in enumerate(self.hidden_sizes):
            self._layers.append(DenseLayer(
                input_size=current_size,
                output_size=hidden_size,
                activation=self.activation,
                name=f"dense_{i}"
            ))

            if self.dropout_rate > 0:
                self._layers.append(DropoutLayer(
                    rate=self.dropout_rate,
                    name=f"dropout_{i}"
                ))

            current_size = hidden_size

        self._layers.append(DenseLayer(
            input_size=current_size,
            output_size=self.output_size,
            activation=ActivationType.LINEAR,
            name="output"
        ))

        logger.info(f"Built MLP model with {len(self._layers)} layers")


def create_lstm_model(
    hidden_sizes: list[int],
    output_size: int = 1,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    name: str = "LSTM"
) -> LSTMModel:
    """
    Factory function to create LSTM model.

    Args:
        hidden_sizes: Hidden layer sizes
        output_size: Output dimension
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        name: Model name

    Returns:
        LSTMModel instance
    """
    config = TrainingConfig(learning_rate=learning_rate)
    return LSTMModel(
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        name=name,
        training_config=config
    )


def create_gru_model(
    hidden_sizes: list[int],
    output_size: int = 1,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    name: str = "GRU"
) -> GRUModel:
    """
    Factory function to create GRU model.

    Args:
        hidden_sizes: Hidden layer sizes
        output_size: Output dimension
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        name: Model name

    Returns:
        GRUModel instance
    """
    config = TrainingConfig(learning_rate=learning_rate)
    return GRUModel(
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        name=name,
        training_config=config
    )


def create_transformer_model(
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    output_size: int = 1,
    learning_rate: float = 0.001,
    name: str = "Transformer"
) -> TransformerModel:
    """
    Factory function to create Transformer model.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of layers
        output_size: Output dimension
        learning_rate: Learning rate
        name: Model name

    Returns:
        TransformerModel instance
    """
    config = TrainingConfig(learning_rate=learning_rate)
    return TransformerModel(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        output_size=output_size,
        name=name,
        training_config=config
    )


def create_mlp_model(
    hidden_sizes: list[int],
    output_size: int = 1,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    name: str = "MLP"
) -> MLPModel:
    """
    Factory function to create MLP model.

    Args:
        hidden_sizes: Hidden layer sizes
        output_size: Output dimension
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        name: Model name

    Returns:
        MLPModel instance
    """
    config = TrainingConfig(learning_rate=learning_rate)
    return MLPModel(
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        dropout_rate=dropout_rate,
        name=name,
        training_config=config
    )
