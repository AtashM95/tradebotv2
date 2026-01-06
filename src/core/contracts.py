
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime

@dataclass
class RunContext:
    run_id: str
    mode: str
    timestamps: Dict[str, datetime]
    cost_budget: float
    dry_run: bool

@dataclass
class MarketSnapshot:
    symbol: str
    price: float
    history: List[float]
    timestamp: datetime

@dataclass
class SignalIntent:
    symbol: str
    action: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskDecision:
    approved: bool
    reason: str
    adjusted_size: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionRequest:
    symbol: str
    action: str
    quantity: float
    price: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TradeFill:
    symbol: str
    action: str
    quantity: float
    price: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
