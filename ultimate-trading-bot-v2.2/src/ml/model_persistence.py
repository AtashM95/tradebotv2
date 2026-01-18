"""
Model Persistence Module for Ultimate Trading Bot v2.2

Provides model serialization, versioning, and storage management
for saving and loading ML models with full state preservation.

Author: AI Assistant
Version: 2.2.0
"""

import gzip
import hashlib
import json
import logging
import pickle
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """Model storage formats."""
    PICKLE = "pickle"
    JSON = "json"
    NUMPY = "numpy"
    BINARY = "binary"


class CompressionType(Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"


@dataclass
class ModelMetadata:
    """Metadata for saved models."""
    model_name: str
    model_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    file_size: int
    checksum: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "file_size": self.file_size,
            "checksum": self.checksum,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "tags": self.tags
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            version=data["version"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            file_size=data["file_size"],
            checksum=data["checksum"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            tags=data.get("tags", [])
        )


@dataclass
class ModelVersion:
    """Version information for a model."""
    major: int
    minor: int
    patch: int
    build: Optional[str] = None

    def __str__(self) -> str:
        """Convert to string."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            version += f"+{self.build}"
        return version

    @classmethod
    def parse(cls, version_str: str) -> "ModelVersion":
        """Parse version string."""
        if "+" in version_str:
            version_part, build = version_str.split("+")
        else:
            version_part = version_str
            build = None

        parts = version_part.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

        return cls(major=major, minor=minor, patch=patch, build=build)

    def increment_patch(self) -> "ModelVersion":
        """Increment patch version."""
        return ModelVersion(self.major, self.minor, self.patch + 1, self.build)

    def increment_minor(self) -> "ModelVersion":
        """Increment minor version."""
        return ModelVersion(self.major, self.minor + 1, 0, self.build)

    def increment_major(self) -> "ModelVersion":
        """Increment major version."""
        return ModelVersion(self.major + 1, 0, 0, self.build)


class BaseSerializer(ABC):
    """Base class for model serializers."""

    def __init__(self, compression: CompressionType = CompressionType.NONE):
        """
        Initialize serializer.

        Args:
            compression: Compression type to use
        """
        self.compression = compression

    @abstractmethod
    def serialize(self, model: Any) -> bytes:
        """Serialize model to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to model."""
        pass

    def _compress(self, data: bytes) -> bytes:
        """Compress data."""
        if self.compression == CompressionType.GZIP:
            return gzip.compress(data)
        return data

    def _decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        if self.compression == CompressionType.GZIP:
            return gzip.decompress(data)
        return data


class PickleSerializer(BaseSerializer):
    """Pickle-based serializer."""

    def serialize(self, model: Any) -> bytes:
        """
        Serialize model using pickle.

        Args:
            model: Model to serialize

        Returns:
            Serialized bytes
        """
        data = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        return self._compress(data)

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize model from pickle bytes.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized model
        """
        data = self._decompress(data)
        return pickle.loads(data)


class JSONSerializer(BaseSerializer):
    """JSON-based serializer for simple models."""

    def serialize(self, model: Any) -> bytes:
        """
        Serialize model to JSON.

        Args:
            model: Model to serialize

        Returns:
            Serialized bytes
        """
        if hasattr(model, "to_dict"):
            data = model.to_dict()
        elif isinstance(model, dict):
            data = model
        else:
            data = self._convert_to_serializable(model)

        json_str = json.dumps(data, indent=2, default=self._json_encoder)
        return self._compress(json_str.encode("utf-8"))

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize model from JSON bytes.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized data (dict)
        """
        data = self._decompress(data)
        return json.loads(data.decode("utf-8"))

    def _json_encoder(self, obj: Any) -> Any:
        """Custom JSON encoder."""
        if isinstance(obj, np.ndarray):
            return {"__numpy__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime):
            return {"__datetime__": True, "value": obj.isoformat()}
        elif hasattr(obj, "to_dict"):
            return obj.to_dict()
        else:
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _convert_to_serializable(self, obj: Any) -> dict[str, Any]:
        """Convert object to serializable dict."""
        if hasattr(obj, "__dict__"):
            return {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__class__.__module__,
                "attributes": {
                    k: self._json_encoder(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
                    else v
                    for k, v in obj.__dict__.items()
                }
            }
        return {}


class NumpySerializer(BaseSerializer):
    """NumPy-based serializer for array-heavy models."""

    def serialize(self, model: Any) -> bytes:
        """
        Serialize model with NumPy arrays.

        Args:
            model: Model to serialize

        Returns:
            Serialized bytes
        """
        arrays = {}
        metadata = {}

        if hasattr(model, "__dict__"):
            for key, value in model.__dict__.items():
                if isinstance(value, np.ndarray):
                    arrays[key] = value
                else:
                    try:
                        metadata[key] = json.dumps(value, default=str)
                    except (TypeError, ValueError):
                        metadata[key] = str(value)

        output = b""

        metadata_json = json.dumps(metadata).encode("utf-8")
        output += struct.pack("I", len(metadata_json))
        output += metadata_json

        output += struct.pack("I", len(arrays))

        for key, arr in arrays.items():
            key_bytes = key.encode("utf-8")
            output += struct.pack("I", len(key_bytes))
            output += key_bytes

            arr_bytes = arr.tobytes()
            output += struct.pack("I", len(arr.shape))
            for dim in arr.shape:
                output += struct.pack("I", dim)

            dtype_str = str(arr.dtype).encode("utf-8")
            output += struct.pack("I", len(dtype_str))
            output += dtype_str

            output += struct.pack("I", len(arr_bytes))
            output += arr_bytes

        return self._compress(output)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """
        Deserialize numpy model.

        Args:
            data: Serialized bytes

        Returns:
            Dictionary with metadata and arrays
        """
        data = self._decompress(data)
        offset = 0

        metadata_len = struct.unpack_from("I", data, offset)[0]
        offset += 4
        metadata_json = data[offset:offset + metadata_len].decode("utf-8")
        metadata = json.loads(metadata_json)
        offset += metadata_len

        n_arrays = struct.unpack_from("I", data, offset)[0]
        offset += 4

        arrays = {}
        for _ in range(n_arrays):
            key_len = struct.unpack_from("I", data, offset)[0]
            offset += 4
            key = data[offset:offset + key_len].decode("utf-8")
            offset += key_len

            n_dims = struct.unpack_from("I", data, offset)[0]
            offset += 4
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack_from("I", data, offset)[0])
                offset += 4

            dtype_len = struct.unpack_from("I", data, offset)[0]
            offset += 4
            dtype_str = data[offset:offset + dtype_len].decode("utf-8")
            offset += dtype_len

            arr_len = struct.unpack_from("I", data, offset)[0]
            offset += 4
            arr_bytes = data[offset:offset + arr_len]
            offset += arr_len

            arrays[key] = np.frombuffer(arr_bytes, dtype=np.dtype(dtype_str)).reshape(shape)

        return {"metadata": metadata, "arrays": arrays}


class ModelStorage:
    """Model storage manager."""

    def __init__(
        self,
        base_path: Union[str, Path],
        default_format: StorageFormat = StorageFormat.PICKLE,
        compression: CompressionType = CompressionType.GZIP
    ):
        """
        Initialize model storage.

        Args:
            base_path: Base directory for model storage
            default_format: Default storage format
            compression: Compression type
        """
        self.base_path = Path(base_path)
        self.default_format = default_format
        self.compression = compression

        self._serializers = {
            StorageFormat.PICKLE: PickleSerializer(compression),
            StorageFormat.JSON: JSONSerializer(compression),
            StorageFormat.NUMPY: NumpySerializer(compression)
        }

        self._ensure_directory()

        logger.info(
            f"Initialized ModelStorage: path={self.base_path}, "
            f"format={default_format.value}"
        )

    def _ensure_directory(self) -> None:
        """Ensure storage directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "checkpoints").mkdir(exist_ok=True)

    def save(
        self,
        model: Any,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        format: Optional[StorageFormat] = None,
        tags: Optional[list[str]] = None
    ) -> ModelMetadata:
        """
        Save model to storage.

        Args:
            model: Model to save
            name: Model name
            version: Model version
            metadata: Additional metadata
            format: Storage format
            tags: Model tags

        Returns:
            ModelMetadata object
        """
        try:
            format = format or self.default_format
            serializer = self._serializers[format]

            model_data = serializer.serialize(model)

            checksum = hashlib.sha256(model_data).hexdigest()

            if version is None:
                existing_version = self._get_latest_version(name)
                if existing_version:
                    version = str(ModelVersion.parse(existing_version).increment_patch())
                else:
                    version = "1.0.0"

            ext = self._get_extension(format)
            model_filename = f"{name}_{version}{ext}"
            model_path = self.base_path / "models" / model_filename

            with open(model_path, "wb") as f:
                f.write(model_data)

            model_type = type(model).__name__ if hasattr(model, "__class__") else "unknown"

            parameters = {}
            if hasattr(model, "__dict__"):
                for key, value in model.__dict__.items():
                    if isinstance(value, (int, float, str, bool)):
                        parameters[key] = value
                    elif isinstance(value, np.ndarray):
                        parameters[key] = f"ndarray{value.shape}"

            model_metadata = ModelMetadata(
                model_name=name,
                model_type=model_type,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_size=len(model_data),
                checksum=checksum,
                parameters=parameters,
                metrics=metadata.get("metrics", {}) if metadata else {},
                tags=tags or []
            )

            metadata_path = self.base_path / "metadata" / f"{name}_{version}.json"
            with open(metadata_path, "w") as f:
                json.dump(model_metadata.to_dict(), f, indent=2)

            logger.info(f"Saved model: {name} v{version} ({len(model_data)} bytes)")

            return model_metadata

        except Exception as e:
            logger.error(f"Error saving model {name}: {e}")
            raise

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        format: Optional[StorageFormat] = None
    ) -> Any:
        """
        Load model from storage.

        Args:
            name: Model name
            version: Model version (latest if not specified)
            format: Storage format

        Returns:
            Loaded model
        """
        try:
            if version is None:
                version = self._get_latest_version(name)
                if version is None:
                    raise FileNotFoundError(f"No versions found for model: {name}")

            format = format or self.default_format
            serializer = self._serializers[format]

            ext = self._get_extension(format)
            model_filename = f"{name}_{version}{ext}"
            model_path = self.base_path / "models" / model_filename

            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            with open(model_path, "rb") as f:
                model_data = f.read()

            metadata = self.get_metadata(name, version)
            if metadata:
                checksum = hashlib.sha256(model_data).hexdigest()
                if checksum != metadata.checksum:
                    logger.warning(f"Checksum mismatch for model {name} v{version}")

            model = serializer.deserialize(model_data)

            logger.info(f"Loaded model: {name} v{version}")

            return model

        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
            raise

    def get_metadata(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Get model metadata.

        Args:
            name: Model name
            version: Model version

        Returns:
            ModelMetadata or None
        """
        if version is None:
            version = self._get_latest_version(name)
            if version is None:
                return None

        metadata_path = self.base_path / "metadata" / f"{name}_{version}.json"

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return ModelMetadata.from_dict(data)

    def list_models(self) -> list[str]:
        """List all saved models."""
        models_dir = self.base_path / "models"
        model_files = list(models_dir.glob("*"))

        model_names = set()
        for path in model_files:
            name_parts = path.stem.rsplit("_", 1)
            if len(name_parts) >= 1:
                model_names.add(name_parts[0])

        return sorted(model_names)

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a model."""
        models_dir = self.base_path / "models"
        pattern = f"{name}_*"
        model_files = list(models_dir.glob(pattern))

        versions = []
        for path in model_files:
            name_parts = path.stem.rsplit("_", 1)
            if len(name_parts) == 2:
                versions.append(name_parts[1])

        return sorted(versions, key=lambda v: ModelVersion.parse(v).major * 10000 +
                     ModelVersion.parse(v).minor * 100 + ModelVersion.parse(v).patch)

    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """
        Delete a model.

        Args:
            name: Model name
            version: Model version (all versions if not specified)

        Returns:
            True if deleted successfully
        """
        try:
            if version:
                versions = [version]
            else:
                versions = self.list_versions(name)

            for v in versions:
                for format in StorageFormat:
                    ext = self._get_extension(format)
                    model_path = self.base_path / "models" / f"{name}_{v}{ext}"
                    if model_path.exists():
                        model_path.unlink()

                metadata_path = self.base_path / "metadata" / f"{name}_{v}.json"
                if metadata_path.exists():
                    metadata_path.unlink()

            logger.info(f"Deleted model: {name} (versions: {versions})")
            return True

        except Exception as e:
            logger.error(f"Error deleting model {name}: {e}")
            return False

    def save_checkpoint(
        self,
        model: Any,
        name: str,
        epoch: int,
        metrics: Optional[dict[str, float]] = None
    ) -> Path:
        """
        Save training checkpoint.

        Args:
            model: Model to checkpoint
            name: Model name
            epoch: Current epoch
            metrics: Training metrics

        Returns:
            Path to checkpoint file
        """
        checkpoint_data = {
            "model": model,
            "epoch": epoch,
            "metrics": metrics or {},
            "timestamp": datetime.now().isoformat()
        }

        serializer = self._serializers[StorageFormat.PICKLE]
        data = serializer.serialize(checkpoint_data)

        checkpoint_filename = f"{name}_epoch{epoch:04d}.ckpt"
        checkpoint_path = self.base_path / "checkpoints" / checkpoint_filename

        with open(checkpoint_path, "wb") as f:
            f.write(data)

        logger.info(f"Saved checkpoint: {checkpoint_filename}")

        return checkpoint_path

    def load_checkpoint(
        self,
        name: str,
        epoch: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            name: Model name
            epoch: Epoch number (latest if not specified)

        Returns:
            Checkpoint data
        """
        checkpoints_dir = self.base_path / "checkpoints"

        if epoch is not None:
            checkpoint_path = checkpoints_dir / f"{name}_epoch{epoch:04d}.ckpt"
        else:
            pattern = f"{name}_epoch*.ckpt"
            checkpoints = list(checkpoints_dir.glob(pattern))

            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found for: {name}")

            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        serializer = self._serializers[StorageFormat.PICKLE]

        with open(checkpoint_path, "rb") as f:
            data = f.read()

        checkpoint = serializer.deserialize(data)

        logger.info(f"Loaded checkpoint: {checkpoint_path.name}")

        return checkpoint

    def _get_latest_version(self, name: str) -> Optional[str]:
        """Get latest version of a model."""
        versions = self.list_versions(name)
        return versions[-1] if versions else None

    def _get_extension(self, format: StorageFormat) -> str:
        """Get file extension for format."""
        extensions = {
            StorageFormat.PICKLE: ".pkl",
            StorageFormat.JSON: ".json",
            StorageFormat.NUMPY: ".npz",
            StorageFormat.BINARY: ".bin"
        }
        ext = extensions.get(format, ".pkl")
        if self.compression == CompressionType.GZIP:
            ext += ".gz"
        return ext


class ModelRegistry:
    """Registry for managing multiple model stores."""

    def __init__(self):
        """Initialize model registry."""
        self._stores: dict[str, ModelStorage] = {}
        self._default_store: Optional[str] = None

        logger.info("Initialized ModelRegistry")

    def register_store(
        self,
        name: str,
        storage: ModelStorage,
        default: bool = False
    ) -> None:
        """
        Register a model storage.

        Args:
            name: Store name
            storage: ModelStorage instance
            default: Whether this is the default store
        """
        self._stores[name] = storage

        if default or self._default_store is None:
            self._default_store = name

        logger.info(f"Registered store: {name}")

    def get_store(self, name: Optional[str] = None) -> ModelStorage:
        """
        Get a model storage.

        Args:
            name: Store name (default if not specified)

        Returns:
            ModelStorage instance
        """
        if name is None:
            name = self._default_store

        if name is None or name not in self._stores:
            raise KeyError(f"Store not found: {name}")

        return self._stores[name]

    def list_stores(self) -> list[str]:
        """List all registered stores."""
        return list(self._stores.keys())


def create_model_storage(
    base_path: str,
    format: str = "pickle",
    compression: str = "gzip"
) -> ModelStorage:
    """
    Factory function to create model storage.

    Args:
        base_path: Base directory path
        format: Storage format
        compression: Compression type

    Returns:
        ModelStorage instance
    """
    return ModelStorage(
        base_path=base_path,
        default_format=StorageFormat(format),
        compression=CompressionType(compression)
    )


def create_model_registry() -> ModelRegistry:
    """
    Factory function to create model registry.

    Returns:
        ModelRegistry instance
    """
    return ModelRegistry()
