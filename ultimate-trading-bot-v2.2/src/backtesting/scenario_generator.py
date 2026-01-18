"""
Scenario Generator for Backtesting.

This module provides scenario generation capabilities for stress testing,
what-if analysis, and synthetic data generation for strategy validation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Types of market scenarios."""

    HISTORICAL = "historical"
    SYNTHETIC = "synthetic"
    STRESS = "stress"
    MONTE_CARLO = "monte_carlo"
    REGIME_BASED = "regime_based"
    CUSTOM = "custom"


class MarketRegime(str, Enum):
    """Market regime types."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRASH = "crash"
    RECOVERY = "recovery"
    NORMAL = "normal"


class StressScenarioType(str, Enum):
    """Types of stress scenarios."""

    MARKET_CRASH = "market_crash"
    FLASH_CRASH = "flash_crash"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    GAP_DOWN = "gap_down"
    GAP_UP = "gap_up"
    EXTENDED_DRAWDOWN = "extended_drawdown"
    WHIPSAW = "whipsaw"


class ScenarioConfig(BaseModel):
    """Configuration for scenario generation."""

    scenario_type: ScenarioType = Field(
        default=ScenarioType.SYNTHETIC,
        description="Type of scenario to generate",
    )
    num_scenarios: int = Field(default=100, description="Number of scenarios")
    scenario_length: int = Field(default=252, description="Length of each scenario in periods")
    random_seed: int | None = Field(default=None, description="Random seed")
    preserve_statistics: bool = Field(default=True, description="Preserve historical statistics")
    include_fat_tails: bool = Field(default=True, description="Include fat-tailed distributions")
    correlation_structure: bool = Field(default=True, description="Preserve correlation structure")
    annualization_factor: float = Field(default=252.0, description="Trading days per year")


@dataclass
class GeneratedScenario:
    """Single generated scenario."""

    scenario_id: int
    scenario_type: ScenarioType
    name: str
    description: str
    data: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)

    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_max_drawdown: float = 0.0
    regime: MarketRegime = MarketRegime.NORMAL


@dataclass
class ScenarioSet:
    """Collection of generated scenarios."""

    scenarios: list[GeneratedScenario]
    config: ScenarioConfig
    base_data: pd.DataFrame | None = None

    generation_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class HistoricalScenarioExtractor:
    """Extract scenarios from historical data."""

    def __init__(
        self,
        lookback_years: int = 20,
    ) -> None:
        """
        Initialize historical scenario extractor.

        Args:
            lookback_years: Years of historical data to consider
        """
        self.lookback_years = lookback_years
        self.known_events = self._initialize_known_events()

    def _initialize_known_events(self) -> dict[str, dict[str, Any]]:
        """Initialize known historical events."""
        return {
            "dot_com_crash": {
                "start": datetime(2000, 3, 10),
                "end": datetime(2002, 10, 9),
                "type": StressScenarioType.EXTENDED_DRAWDOWN,
                "description": "Dot-com bubble burst",
            },
            "financial_crisis_2008": {
                "start": datetime(2008, 9, 15),
                "end": datetime(2009, 3, 9),
                "type": StressScenarioType.MARKET_CRASH,
                "description": "Global Financial Crisis",
            },
            "flash_crash_2010": {
                "start": datetime(2010, 5, 6),
                "end": datetime(2010, 5, 6),
                "type": StressScenarioType.FLASH_CRASH,
                "description": "Flash Crash of 2010",
            },
            "covid_crash_2020": {
                "start": datetime(2020, 2, 20),
                "end": datetime(2020, 3, 23),
                "type": StressScenarioType.MARKET_CRASH,
                "description": "COVID-19 Market Crash",
            },
            "vix_spike_2018": {
                "start": datetime(2018, 2, 2),
                "end": datetime(2018, 2, 9),
                "type": StressScenarioType.VOLATILITY_SPIKE,
                "description": "VIX Spike February 2018",
            },
        }

    def extract_scenario(
        self,
        data: pd.DataFrame,
        event_name: str,
    ) -> GeneratedScenario | None:
        """
        Extract a specific historical scenario.

        Args:
            data: Historical data
            event_name: Name of the event to extract

        Returns:
            Extracted scenario or None if not found
        """
        if event_name not in self.known_events:
            logger.warning(f"Unknown event: {event_name}")
            return None

        event = self.known_events[event_name]
        start = event["start"]
        end = event["end"]

        try:
            scenario_data = data.loc[start:end].copy()

            if len(scenario_data) == 0:
                logger.warning(f"No data available for event: {event_name}")
                return None

            returns = scenario_data["close"].pct_change().dropna()

            return GeneratedScenario(
                scenario_id=0,
                scenario_type=ScenarioType.HISTORICAL,
                name=event_name,
                description=event["description"],
                data=scenario_data,
                metadata={"event_type": event["type"].value},
                start_date=scenario_data.index.min().to_pydatetime(),
                end_date=scenario_data.index.max().to_pydatetime(),
                expected_return=float(np.prod(1 + returns) - 1),
                expected_volatility=float(returns.std() * np.sqrt(252)),
                expected_max_drawdown=self._calculate_max_drawdown(scenario_data["close"]),
                regime=MarketRegime.BEAR if returns.sum() < 0 else MarketRegime.BULL,
            )

        except Exception as e:
            logger.error(f"Error extracting scenario {event_name}: {e}")
            return None

    def extract_worst_periods(
        self,
        data: pd.DataFrame,
        num_periods: int = 5,
        period_length: int = 21,
    ) -> list[GeneratedScenario]:
        """
        Extract worst performing periods from historical data.

        Args:
            data: Historical data
            num_periods: Number of worst periods to extract
            period_length: Length of each period in days

        Returns:
            List of worst period scenarios
        """
        scenarios = []

        if "close" not in data.columns:
            logger.error("Data must contain 'close' column")
            return scenarios

        returns = data["close"].pct_change().dropna()

        rolling_returns = returns.rolling(window=period_length).sum()

        worst_ends = rolling_returns.nsmallest(num_periods * 2).index.tolist()

        used_dates: set[datetime] = set()
        scenario_id = 0

        for end_date in worst_ends:
            if len(scenarios) >= num_periods:
                break

            end_idx = data.index.get_loc(end_date)
            start_idx = max(0, end_idx - period_length + 1)
            start_date = data.index[start_idx]

            overlap = False
            for used_date in used_dates:
                if abs((end_date - used_date).days) < period_length:
                    overlap = True
                    break

            if overlap:
                continue

            used_dates.add(end_date)

            period_data = data.iloc[start_idx : end_idx + 1].copy()
            period_returns = period_data["close"].pct_change().dropna()

            scenarios.append(
                GeneratedScenario(
                    scenario_id=scenario_id,
                    scenario_type=ScenarioType.HISTORICAL,
                    name=f"worst_period_{scenario_id + 1}",
                    description=f"Historical worst period ending {end_date.date()}",
                    data=period_data,
                    metadata={
                        "period_return": float(rolling_returns.loc[end_date]),
                        "rank": scenario_id + 1,
                    },
                    start_date=start_date.to_pydatetime(),
                    end_date=end_date.to_pydatetime(),
                    expected_return=float(np.prod(1 + period_returns) - 1),
                    expected_volatility=float(period_returns.std() * np.sqrt(252)),
                    expected_max_drawdown=self._calculate_max_drawdown(period_data["close"]),
                    regime=MarketRegime.BEAR,
                )
            )
            scenario_id += 1

        return scenarios

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = prices.expanding().max()
        drawdowns = (prices - running_max) / running_max
        return float(drawdowns.min())


class SyntheticScenarioGenerator:
    """Generate synthetic market scenarios."""

    def __init__(
        self,
        config: ScenarioConfig | None = None,
    ) -> None:
        """
        Initialize synthetic scenario generator.

        Args:
            config: Generation configuration
        """
        self.config = config or ScenarioConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def generate_geometric_brownian_motion(
        self,
        mu: float = 0.10,
        sigma: float = 0.20,
        s0: float = 100.0,
        num_scenarios: int | None = None,
        scenario_length: int | None = None,
    ) -> list[GeneratedScenario]:
        """
        Generate scenarios using Geometric Brownian Motion.

        Args:
            mu: Annual drift (expected return)
            sigma: Annual volatility
            s0: Initial price
            num_scenarios: Number of scenarios to generate
            scenario_length: Length of each scenario

        Returns:
            List of generated scenarios
        """
        n_scenarios = num_scenarios or self.config.num_scenarios
        n_periods = scenario_length or self.config.scenario_length

        scenarios = []
        dt = 1.0 / self.config.annualization_factor

        for i in range(n_scenarios):
            dW = np.random.normal(0, np.sqrt(dt), size=n_periods)

            log_returns = (mu - 0.5 * sigma**2) * dt + sigma * dW

            prices = s0 * np.exp(np.cumsum(log_returns))
            prices = np.insert(prices, 0, s0)

            dates = pd.date_range(
                start=datetime.now(),
                periods=len(prices),
                freq="D",
            )

            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                    "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(prices)),
                },
                index=dates,
            )

            returns = df["close"].pct_change().dropna()

            scenarios.append(
                GeneratedScenario(
                    scenario_id=i,
                    scenario_type=ScenarioType.SYNTHETIC,
                    name=f"gbm_scenario_{i}",
                    description=f"GBM scenario with mu={mu:.2%}, sigma={sigma:.2%}",
                    data=df,
                    metadata={"mu": mu, "sigma": sigma, "s0": s0},
                    start_date=dates[0].to_pydatetime(),
                    end_date=dates[-1].to_pydatetime(),
                    expected_return=float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
                    expected_volatility=float(returns.std() * np.sqrt(252)) if len(returns) > 0 else sigma,
                    expected_max_drawdown=self._calculate_max_drawdown(df["close"]),
                    regime=MarketRegime.BULL if mu > 0 else MarketRegime.BEAR,
                )
            )

        return scenarios

    def generate_regime_switching(
        self,
        regimes: list[dict[str, Any]] | None = None,
        transition_matrix: np.ndarray | None = None,
        num_scenarios: int | None = None,
        scenario_length: int | None = None,
    ) -> list[GeneratedScenario]:
        """
        Generate scenarios with regime switching dynamics.

        Args:
            regimes: List of regime parameters
            transition_matrix: Markov transition matrix
            num_scenarios: Number of scenarios
            scenario_length: Length of each scenario

        Returns:
            List of generated scenarios
        """
        if regimes is None:
            regimes = [
                {"name": "bull", "mu": 0.15, "sigma": 0.12},
                {"name": "bear", "mu": -0.10, "sigma": 0.25},
                {"name": "normal", "mu": 0.08, "sigma": 0.18},
            ]

        n_regimes = len(regimes)

        if transition_matrix is None:
            transition_matrix = np.array([
                [0.95, 0.03, 0.02],
                [0.05, 0.90, 0.05],
                [0.03, 0.02, 0.95],
            ])

        n_scenarios = num_scenarios or self.config.num_scenarios
        n_periods = scenario_length or self.config.scenario_length

        scenarios = []
        dt = 1.0 / self.config.annualization_factor

        for i in range(n_scenarios):
            current_regime = np.random.randint(0, n_regimes)
            regime_sequence = [current_regime]

            for _ in range(n_periods - 1):
                current_regime = np.random.choice(
                    n_regimes,
                    p=transition_matrix[current_regime],
                )
                regime_sequence.append(current_regime)

            returns = []
            for regime_idx in regime_sequence:
                regime = regimes[regime_idx]
                r = np.random.normal(
                    regime["mu"] * dt,
                    regime["sigma"] * np.sqrt(dt),
                )
                returns.append(r)

            prices = 100.0 * np.cumprod(1 + np.array(returns))
            prices = np.insert(prices, 0, 100.0)

            dates = pd.date_range(
                start=datetime.now(),
                periods=len(prices),
                freq="D",
            )

            df = pd.DataFrame(
                {
                    "open": prices,
                    "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                    "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                    "close": prices,
                    "volume": np.random.randint(1000000, 10000000, len(prices)),
                    "regime": [0] + regime_sequence,
                },
                index=dates,
            )

            ret_series = df["close"].pct_change().dropna()

            scenarios.append(
                GeneratedScenario(
                    scenario_id=i,
                    scenario_type=ScenarioType.REGIME_BASED,
                    name=f"regime_switch_scenario_{i}",
                    description="Regime-switching scenario",
                    data=df,
                    metadata={
                        "regimes": regimes,
                        "regime_sequence": regime_sequence,
                    },
                    start_date=dates[0].to_pydatetime(),
                    end_date=dates[-1].to_pydatetime(),
                    expected_return=float(np.prod(1 + ret_series) - 1) if len(ret_series) > 0 else 0.0,
                    expected_volatility=float(ret_series.std() * np.sqrt(252)) if len(ret_series) > 0 else 0.2,
                    expected_max_drawdown=self._calculate_max_drawdown(df["close"]),
                    regime=MarketRegime.NORMAL,
                )
            )

        return scenarios

    def generate_stress_scenario(
        self,
        stress_type: StressScenarioType,
        base_price: float = 100.0,
        scenario_length: int | None = None,
    ) -> GeneratedScenario:
        """
        Generate a specific stress scenario.

        Args:
            stress_type: Type of stress scenario
            base_price: Starting price
            scenario_length: Length of scenario

        Returns:
            Generated stress scenario
        """
        n_periods = scenario_length or min(63, self.config.scenario_length)

        if stress_type == StressScenarioType.MARKET_CRASH:
            returns = self._generate_crash_returns(n_periods, severity=-0.40)
        elif stress_type == StressScenarioType.FLASH_CRASH:
            returns = self._generate_flash_crash_returns(n_periods)
        elif stress_type == StressScenarioType.VOLATILITY_SPIKE:
            returns = self._generate_vol_spike_returns(n_periods)
        elif stress_type == StressScenarioType.GAP_DOWN:
            returns = self._generate_gap_returns(n_periods, direction=-1)
        elif stress_type == StressScenarioType.GAP_UP:
            returns = self._generate_gap_returns(n_periods, direction=1)
        elif stress_type == StressScenarioType.WHIPSAW:
            returns = self._generate_whipsaw_returns(n_periods)
        elif stress_type == StressScenarioType.EXTENDED_DRAWDOWN:
            returns = self._generate_extended_drawdown_returns(n_periods)
        else:
            returns = self._generate_crash_returns(n_periods, severity=-0.30)

        prices = base_price * np.cumprod(1 + np.array(returns))
        prices = np.insert(prices, 0, base_price)

        dates = pd.date_range(
            start=datetime.now(),
            periods=len(prices),
            freq="D",
        )

        daily_range = np.abs(np.random.normal(0, 0.02, len(prices)))

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + daily_range),
                "low": prices * (1 - daily_range),
                "close": prices,
                "volume": np.random.randint(5000000, 50000000, len(prices)),
            },
            index=dates,
        )

        ret_series = pd.Series(returns)

        return GeneratedScenario(
            scenario_id=0,
            scenario_type=ScenarioType.STRESS,
            name=f"stress_{stress_type.value}",
            description=f"Stress scenario: {stress_type.value}",
            data=df,
            metadata={"stress_type": stress_type.value},
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime(),
            expected_return=float(np.prod(1 + ret_series) - 1),
            expected_volatility=float(ret_series.std() * np.sqrt(252)),
            expected_max_drawdown=self._calculate_max_drawdown(df["close"]),
            regime=MarketRegime.CRASH,
        )

    def _generate_crash_returns(
        self,
        n_periods: int,
        severity: float = -0.40,
    ) -> list[float]:
        """Generate crash scenario returns."""
        returns = []
        remaining_drop = severity

        for i in range(n_periods):
            if i < n_periods // 3:
                daily_drop = remaining_drop / (n_periods // 3) * np.random.uniform(0.5, 1.5)
                remaining_drop -= daily_drop
                r = daily_drop + np.random.normal(0, 0.02)
            else:
                r = np.random.normal(0, 0.025)

            returns.append(r)

        return returns

    def _generate_flash_crash_returns(self, n_periods: int) -> list[float]:
        """Generate flash crash returns."""
        returns = []
        crash_day = np.random.randint(5, n_periods // 2)

        for i in range(n_periods):
            if i == crash_day:
                r = np.random.uniform(-0.08, -0.05)
            elif i == crash_day + 1:
                r = np.random.uniform(0.03, 0.06)
            else:
                r = np.random.normal(0, 0.015)

            returns.append(r)

        return returns

    def _generate_vol_spike_returns(self, n_periods: int) -> list[float]:
        """Generate volatility spike returns."""
        returns = []
        spike_start = np.random.randint(5, n_periods // 2)
        spike_duration = np.random.randint(5, 15)

        for i in range(n_periods):
            if spike_start <= i < spike_start + spike_duration:
                r = np.random.normal(0, 0.05)
            else:
                r = np.random.normal(0, 0.01)

            returns.append(r)

        return returns

    def _generate_gap_returns(
        self,
        n_periods: int,
        direction: int = -1,
    ) -> list[float]:
        """Generate gap scenario returns."""
        returns = []
        gap_day = np.random.randint(5, n_periods // 2)
        gap_size = direction * np.random.uniform(0.05, 0.10)

        for i in range(n_periods):
            if i == gap_day:
                r = gap_size
            else:
                r = np.random.normal(0, 0.012)

            returns.append(r)

        return returns

    def _generate_whipsaw_returns(self, n_periods: int) -> list[float]:
        """Generate whipsaw returns."""
        returns = []
        cycle_length = np.random.randint(3, 7)

        for i in range(n_periods):
            cycle_pos = i % (cycle_length * 2)
            if cycle_pos < cycle_length:
                base = 0.015
            else:
                base = -0.015

            r = base + np.random.normal(0, 0.01)
            returns.append(r)

        return returns

    def _generate_extended_drawdown_returns(self, n_periods: int) -> list[float]:
        """Generate extended drawdown returns."""
        returns = []
        daily_drift = -0.0015

        for _ in range(n_periods):
            r = daily_drift + np.random.normal(0, 0.015)
            returns.append(r)

        return returns

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = prices.expanding().max()
        drawdowns = (prices - running_max) / running_max
        return float(drawdowns.min())


class ScenarioGenerator:
    """Main scenario generator combining all methods."""

    def __init__(
        self,
        config: ScenarioConfig | None = None,
    ) -> None:
        """
        Initialize scenario generator.

        Args:
            config: Generation configuration
        """
        self.config = config or ScenarioConfig()
        self.historical_extractor = HistoricalScenarioExtractor()
        self.synthetic_generator = SyntheticScenarioGenerator(config=self.config)

        logger.info("ScenarioGenerator initialized")

    def generate_scenario_set(
        self,
        base_data: pd.DataFrame | None = None,
        include_stress: bool = True,
        include_historical: bool = True,
        include_synthetic: bool = True,
    ) -> ScenarioSet:
        """
        Generate comprehensive scenario set.

        Args:
            base_data: Optional historical data for reference
            include_stress: Include stress scenarios
            include_historical: Include historical scenarios
            include_synthetic: Include synthetic scenarios

        Returns:
            Complete scenario set
        """
        start_time = datetime.now()
        all_scenarios: list[GeneratedScenario] = []
        scenario_id = 0

        if include_synthetic:
            gbm_scenarios = self.synthetic_generator.generate_geometric_brownian_motion(
                mu=0.10,
                sigma=0.20,
                num_scenarios=self.config.num_scenarios // 3,
            )
            for s in gbm_scenarios:
                s.scenario_id = scenario_id
                scenario_id += 1
            all_scenarios.extend(gbm_scenarios)

            regime_scenarios = self.synthetic_generator.generate_regime_switching(
                num_scenarios=self.config.num_scenarios // 3,
            )
            for s in regime_scenarios:
                s.scenario_id = scenario_id
                scenario_id += 1
            all_scenarios.extend(regime_scenarios)

        if include_stress:
            stress_types = [
                StressScenarioType.MARKET_CRASH,
                StressScenarioType.FLASH_CRASH,
                StressScenarioType.VOLATILITY_SPIKE,
                StressScenarioType.GAP_DOWN,
                StressScenarioType.WHIPSAW,
                StressScenarioType.EXTENDED_DRAWDOWN,
            ]

            for stress_type in stress_types:
                stress_scenario = self.synthetic_generator.generate_stress_scenario(
                    stress_type=stress_type,
                )
                stress_scenario.scenario_id = scenario_id
                scenario_id += 1
                all_scenarios.append(stress_scenario)

        if include_historical and base_data is not None:
            for event_name in self.historical_extractor.known_events.keys():
                historical_scenario = self.historical_extractor.extract_scenario(
                    data=base_data,
                    event_name=event_name,
                )
                if historical_scenario:
                    historical_scenario.scenario_id = scenario_id
                    scenario_id += 1
                    all_scenarios.append(historical_scenario)

            worst_periods = self.historical_extractor.extract_worst_periods(
                data=base_data,
                num_periods=5,
            )
            for wp in worst_periods:
                wp.scenario_id = scenario_id
                scenario_id += 1
            all_scenarios.extend(worst_periods)

        elapsed = (datetime.now() - start_time).total_seconds()

        return ScenarioSet(
            scenarios=all_scenarios,
            config=self.config,
            base_data=base_data,
            generation_time=elapsed,
        )

    def generate_custom_scenario(
        self,
        return_generator: Callable[[int], np.ndarray],
        name: str,
        description: str,
        scenario_length: int | None = None,
        base_price: float = 100.0,
    ) -> GeneratedScenario:
        """
        Generate scenario with custom return generator.

        Args:
            return_generator: Function that generates returns
            name: Scenario name
            description: Scenario description
            scenario_length: Length of scenario
            base_price: Starting price

        Returns:
            Custom generated scenario
        """
        n_periods = scenario_length or self.config.scenario_length

        returns = return_generator(n_periods)

        prices = base_price * np.cumprod(1 + returns)
        prices = np.insert(prices, 0, base_price)

        dates = pd.date_range(
            start=datetime.now(),
            periods=len(prices),
            freq="D",
        )

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, len(prices)),
            },
            index=dates,
        )

        ret_series = pd.Series(returns)

        return GeneratedScenario(
            scenario_id=0,
            scenario_type=ScenarioType.CUSTOM,
            name=name,
            description=description,
            data=df,
            metadata={"custom": True},
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime(),
            expected_return=float(np.prod(1 + ret_series) - 1),
            expected_volatility=float(ret_series.std() * np.sqrt(252)),
            expected_max_drawdown=float(
                ((df["close"] - df["close"].expanding().max()) / df["close"].expanding().max()).min()
            ),
            regime=MarketRegime.NORMAL,
        )


def create_scenario_generator(
    num_scenarios: int = 100,
    scenario_length: int = 252,
    config: dict | None = None,
) -> ScenarioGenerator:
    """
    Create a scenario generator.

    Args:
        num_scenarios: Number of scenarios to generate
        scenario_length: Length of each scenario
        config: Additional configuration

    Returns:
        Configured ScenarioGenerator
    """
    gen_config = ScenarioConfig(
        num_scenarios=num_scenarios,
        scenario_length=scenario_length,
        **(config or {}),
    )
    return ScenarioGenerator(config=gen_config)
