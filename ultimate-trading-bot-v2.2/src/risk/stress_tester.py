"""
Stress Testing Framework for Ultimate Trading Bot v2.2.

This module provides comprehensive stress testing capabilities including
historical scenarios, Monte Carlo simulation, and sensitivity analysis.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Types of stress scenarios."""

    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY = "sensitivity"
    REVERSE = "reverse"


class HistoricalEvent(str, Enum):
    """Historical market events for stress testing."""

    BLACK_MONDAY_1987 = "black_monday_1987"
    DOT_COM_CRASH = "dot_com_crash"
    FINANCIAL_CRISIS_2008 = "financial_crisis_2008"
    FLASH_CRASH_2010 = "flash_crash_2010"
    COVID_CRASH_2020 = "covid_crash_2020"
    VOLMAGEDDON_2018 = "volmageddon_2018"
    TAPER_TANTRUM_2013 = "taper_tantrum_2013"
    EURO_CRISIS_2011 = "euro_crisis_2011"
    CHINA_DEVALUATION_2015 = "china_devaluation_2015"
    CUSTOM = "custom"


class StressTesterConfig(BaseModel):
    """Configuration for stress testing."""

    model_config = {"arbitrary_types_allowed": True}

    num_monte_carlo_sims: int = Field(default=10000, description="Monte Carlo simulations")
    confidence_levels: list[float] = Field(
        default_factory=lambda: [0.95, 0.99, 0.999],
        description="Confidence levels for stress VaR"
    )
    sensitivity_shock_sizes: list[float] = Field(
        default_factory=lambda: [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20],
        description="Shock sizes for sensitivity analysis"
    )
    correlation_stress_factor: float = Field(
        default=1.5,
        description="Correlation increase factor during stress"
    )
    volatility_stress_factor: float = Field(default=2.0, description="Vol multiplier in stress")
    liquidity_haircut: float = Field(default=0.1, description="Liquidity discount in stress")
    recovery_period_days: int = Field(default=60, description="Recovery period assumption")
    max_parallel_sims: int = Field(default=100, description="Max parallel simulations")


class StressScenario(BaseModel):
    """Definition of a stress scenario."""

    name: str
    scenario_type: ScenarioType
    description: str = ""

    equity_shock: float = Field(default=0.0, description="Equity market shock")
    volatility_shock: float = Field(default=0.0, description="Volatility shock")
    interest_rate_shock: float = Field(default=0.0, description="Interest rate shock bps")
    credit_spread_shock: float = Field(default=0.0, description="Credit spread shock bps")
    fx_shock: float = Field(default=0.0, description="FX shock")
    commodity_shock: float = Field(default=0.0, description="Commodity shock")

    correlation_adjustment: float = Field(default=0.0, description="Correlation shift")
    liquidity_haircut: float = Field(default=0.0, description="Liquidity discount")
    duration_days: int = Field(default=1, description="Scenario duration")

    custom_shocks: dict[str, float] = Field(default_factory=dict)


class StressTestResult(BaseModel):
    """Results from a stress test."""

    scenario: StressScenario
    timestamp: datetime = Field(default_factory=datetime.now)

    portfolio_loss: float = Field(default=0.0, description="Portfolio loss amount")
    portfolio_loss_pct: float = Field(default=0.0, description="Portfolio loss %")

    position_losses: dict[str, float] = Field(
        default_factory=dict,
        description="Loss by position"
    )
    sector_losses: dict[str, float] = Field(
        default_factory=dict,
        description="Loss by sector"
    )

    worst_position: str = Field(default="", description="Worst performing position")
    worst_position_loss: float = Field(default=0.0, description="Worst position loss")

    margin_impact: float = Field(default=0.0, description="Margin requirement change")
    liquidity_impact: float = Field(default=0.0, description="Liquidity impact")

    recovery_estimate_days: int = Field(default=0, description="Estimated recovery time")

    stress_var: float = Field(default=0.0, description="Stress VaR")
    stress_cvar: float = Field(default=0.0, description="Stress CVaR")

    passed: bool = Field(default=True, description="Whether portfolio survives")
    breach_limits: list[str] = Field(default_factory=list, description="Breached limits")


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo stress simulation."""

    num_simulations: int
    simulation_returns: list[float] = field(default_factory=list)

    mean_loss: float = 0.0
    median_loss: float = 0.0
    std_loss: float = 0.0

    var_95: float = 0.0
    var_99: float = 0.0
    var_999: float = 0.0

    cvar_95: float = 0.0
    cvar_99: float = 0.0

    worst_case: float = 0.0
    best_case: float = 0.0

    probability_of_ruin: float = 0.0
    expected_shortfall: float = 0.0


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis."""

    factor: str
    shock_sizes: list[float] = field(default_factory=list)
    portfolio_impacts: list[float] = field(default_factory=list)
    delta: float = 0.0
    gamma: float = 0.0
    is_linear: bool = True


HISTORICAL_SCENARIOS: dict[HistoricalEvent, StressScenario] = {
    HistoricalEvent.BLACK_MONDAY_1987: StressScenario(
        name="Black Monday 1987",
        scenario_type=ScenarioType.HISTORICAL,
        description="October 19, 1987 - Single day 22.6% market crash",
        equity_shock=-0.226,
        volatility_shock=3.0,
        correlation_adjustment=0.3,
        liquidity_haircut=0.15,
        duration_days=1,
    ),
    HistoricalEvent.DOT_COM_CRASH: StressScenario(
        name="Dot-Com Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="2000-2002 tech bubble burst",
        equity_shock=-0.45,
        volatility_shock=1.5,
        duration_days=500,
    ),
    HistoricalEvent.FINANCIAL_CRISIS_2008: StressScenario(
        name="2008 Financial Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="Global financial crisis peak stress",
        equity_shock=-0.50,
        volatility_shock=4.0,
        credit_spread_shock=500,
        correlation_adjustment=0.4,
        liquidity_haircut=0.25,
        duration_days=90,
    ),
    HistoricalEvent.FLASH_CRASH_2010: StressScenario(
        name="Flash Crash 2010",
        scenario_type=ScenarioType.HISTORICAL,
        description="May 6, 2010 flash crash",
        equity_shock=-0.09,
        volatility_shock=5.0,
        liquidity_haircut=0.30,
        duration_days=1,
    ),
    HistoricalEvent.COVID_CRASH_2020: StressScenario(
        name="COVID-19 Crash",
        scenario_type=ScenarioType.HISTORICAL,
        description="March 2020 pandemic crash",
        equity_shock=-0.34,
        volatility_shock=6.0,
        credit_spread_shock=300,
        correlation_adjustment=0.35,
        liquidity_haircut=0.20,
        duration_days=23,
    ),
    HistoricalEvent.VOLMAGEDDON_2018: StressScenario(
        name="Volmageddon 2018",
        scenario_type=ScenarioType.HISTORICAL,
        description="February 2018 volatility spike",
        equity_shock=-0.10,
        volatility_shock=8.0,
        duration_days=5,
    ),
    HistoricalEvent.TAPER_TANTRUM_2013: StressScenario(
        name="Taper Tantrum 2013",
        scenario_type=ScenarioType.HISTORICAL,
        description="Fed taper announcement reaction",
        equity_shock=-0.06,
        interest_rate_shock=100,
        duration_days=60,
    ),
    HistoricalEvent.EURO_CRISIS_2011: StressScenario(
        name="European Debt Crisis",
        scenario_type=ScenarioType.HISTORICAL,
        description="2011 European sovereign debt crisis",
        equity_shock=-0.20,
        credit_spread_shock=400,
        fx_shock=-0.10,
        duration_days=120,
    ),
    HistoricalEvent.CHINA_DEVALUATION_2015: StressScenario(
        name="China Devaluation 2015",
        scenario_type=ScenarioType.HISTORICAL,
        description="August 2015 China currency devaluation",
        equity_shock=-0.12,
        fx_shock=0.05,
        commodity_shock=-0.15,
        duration_days=30,
    ),
}


class StressTester:
    """
    Performs comprehensive stress testing on portfolios.

    Provides historical scenario analysis, Monte Carlo simulation,
    sensitivity analysis, and reverse stress testing.
    """

    def __init__(self, config: StressTesterConfig | None = None):
        """
        Initialize stress tester.

        Args:
            config: Stress testing configuration
        """
        self.config = config or StressTesterConfig()
        self._custom_scenarios: dict[str, StressScenario] = {}
        self._test_history: list[StressTestResult] = []
        self._lock = asyncio.Lock()

        logger.info("StressTester initialized")

    async def run_historical_stress_test(
        self,
        event: HistoricalEvent,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> StressTestResult:
        """
        Run stress test based on historical event.

        Args:
            event: Historical event to simulate
            portfolio: Portfolio information
            positions: Current positions

        Returns:
            StressTestResult object
        """
        scenario = HISTORICAL_SCENARIOS.get(event)
        if not scenario:
            raise ValueError(f"Unknown historical event: {event}")

        return await self.run_stress_test(scenario, portfolio, positions)

    async def run_stress_test(
        self,
        scenario: StressScenario,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> StressTestResult:
        """
        Run stress test with given scenario.

        Args:
            scenario: Stress scenario to apply
            portfolio: Portfolio information
            positions: Current positions

        Returns:
            StressTestResult object
        """
        try:
            portfolio_value = float(portfolio.get("equity", 0))
            margin_used = float(portfolio.get("margin_used", 0))

            position_losses: dict[str, float] = {}
            sector_losses: dict[str, float] = {}
            total_loss = 0.0

            for pos in positions:
                symbol = pos.get("symbol", "")
                market_value = float(pos.get("market_value", 0))
                sector = pos.get("sector", "Unknown")
                beta = float(pos.get("beta", 1.0))
                is_short = float(pos.get("qty", 0)) < 0

                position_shock = scenario.equity_shock * beta

                if scenario.volatility_shock > 1:
                    vol_impact = (scenario.volatility_shock - 1) * 0.05
                    position_shock -= vol_impact

                if is_short:
                    position_shock = -position_shock

                if scenario.liquidity_haircut > 0:
                    position_shock -= scenario.liquidity_haircut

                position_loss = market_value * position_shock
                position_losses[symbol] = position_loss
                total_loss += position_loss

                if sector in sector_losses:
                    sector_losses[sector] += position_loss
                else:
                    sector_losses[sector] = position_loss

            loss_pct = total_loss / portfolio_value if portfolio_value > 0 else 0.0

            worst_position = ""
            worst_loss = 0.0
            for symbol, loss in position_losses.items():
                if loss < worst_loss:
                    worst_loss = loss
                    worst_position = symbol

            margin_impact = abs(total_loss) * 0.5

            liquidity_impact = abs(total_loss) * scenario.liquidity_haircut

            daily_recovery_rate = 0.01
            if loss_pct < 0:
                recovery_days = int(abs(loss_pct) / daily_recovery_rate)
            else:
                recovery_days = 0

            breach_limits: list[str] = []
            if loss_pct < -0.20:
                breach_limits.append("Maximum drawdown limit (20%)")
            if margin_used + margin_impact > portfolio_value:
                breach_limits.append("Margin requirement")

            passed = len(breach_limits) == 0 and (portfolio_value + total_loss) > 0

            result = StressTestResult(
                scenario=scenario,
                portfolio_loss=total_loss,
                portfolio_loss_pct=loss_pct,
                position_losses=position_losses,
                sector_losses=sector_losses,
                worst_position=worst_position,
                worst_position_loss=worst_loss,
                margin_impact=margin_impact,
                liquidity_impact=liquidity_impact,
                recovery_estimate_days=min(recovery_days, 365),
                stress_var=abs(total_loss),
                stress_cvar=abs(total_loss) * 1.2,
                passed=passed,
                breach_limits=breach_limits,
            )

            async with self._lock:
                self._test_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return StressTestResult(
                scenario=scenario,
                passed=False,
                breach_limits=[f"Test error: {str(e)}"],
            )

    async def run_monte_carlo_stress(
        self,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
        base_volatility: float,
        num_simulations: int | None = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo stress simulation.

        Args:
            portfolio: Portfolio information
            positions: Current positions
            base_volatility: Base portfolio volatility
            num_simulations: Number of simulations

        Returns:
            MonteCarloResult object
        """
        try:
            n_sims = num_simulations or self.config.num_monte_carlo_sims
            portfolio_value = float(portfolio.get("equity", 0))

            stressed_vol = base_volatility * self.config.volatility_stress_factor

            daily_vol = stressed_vol / np.sqrt(252)

            returns = np.random.normal(
                loc=-0.001,
                scale=daily_vol,
                size=n_sims,
            )

            fat_tail_mask = np.random.random(n_sims) < 0.05
            returns[fat_tail_mask] *= 2.0

            losses = returns * portfolio_value

            result = MonteCarloResult(
                num_simulations=n_sims,
                simulation_returns=returns.tolist(),
                mean_loss=float(np.mean(losses)),
                median_loss=float(np.median(losses)),
                std_loss=float(np.std(losses)),
                var_95=float(np.percentile(losses, 5)),
                var_99=float(np.percentile(losses, 1)),
                var_999=float(np.percentile(losses, 0.1)),
                cvar_95=float(np.mean(losses[losses <= np.percentile(losses, 5)])),
                cvar_99=float(np.mean(losses[losses <= np.percentile(losses, 1)])),
                worst_case=float(np.min(losses)),
                best_case=float(np.max(losses)),
                probability_of_ruin=float(np.mean(losses < -portfolio_value * 0.5)),
                expected_shortfall=float(np.mean(losses[losses < 0])),
            )

            return result

        except Exception as e:
            logger.error(f"Monte Carlo stress test failed: {e}")
            return MonteCarloResult(num_simulations=0)

    async def run_sensitivity_analysis(
        self,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
        factor: str = "equity",
    ) -> SensitivityResult:
        """
        Run sensitivity analysis on a risk factor.

        Args:
            portfolio: Portfolio information
            positions: Current positions
            factor: Risk factor to analyze

        Returns:
            SensitivityResult object
        """
        try:
            portfolio_value = float(portfolio.get("equity", 0))
            shock_sizes = self.config.sensitivity_shock_sizes
            impacts: list[float] = []

            for shock in shock_sizes:
                scenario = StressScenario(
                    name=f"Sensitivity_{factor}_{shock}",
                    scenario_type=ScenarioType.SENSITIVITY,
                    equity_shock=shock if factor == "equity" else 0.0,
                    volatility_shock=1 + shock if factor == "volatility" else 1.0,
                    interest_rate_shock=shock * 100 if factor == "interest_rate" else 0.0,
                )

                result = await self.run_stress_test(scenario, portfolio, positions)
                impacts.append(result.portfolio_loss)

            delta = 0.0
            gamma = 0.0

            if len(shock_sizes) >= 2 and len(impacts) >= 2:
                shock_arr = np.array(shock_sizes)
                impact_arr = np.array(impacts)

                if len(shock_sizes) >= 3:
                    coeffs = np.polyfit(shock_arr, impact_arr, 2)
                    gamma = coeffs[0] * 2
                    delta = coeffs[1]
                else:
                    delta = (impacts[1] - impacts[0]) / (shock_sizes[1] - shock_sizes[0])

            residuals = np.array(impacts) - (delta * np.array(shock_sizes))
            is_linear = np.std(residuals) / (np.std(impacts) + 1e-10) < 0.1

            return SensitivityResult(
                factor=factor,
                shock_sizes=shock_sizes,
                portfolio_impacts=impacts,
                delta=delta,
                gamma=gamma,
                is_linear=is_linear,
            )

        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return SensitivityResult(factor=factor)

    async def run_reverse_stress_test(
        self,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
        target_loss_pct: float = 0.25,
    ) -> dict[str, Any]:
        """
        Run reverse stress test to find scenarios causing target loss.

        Args:
            portfolio: Portfolio information
            positions: Current positions
            target_loss_pct: Target portfolio loss percentage

        Returns:
            Dictionary with scenarios causing target loss
        """
        try:
            portfolio_value = float(portfolio.get("equity", 0))
            target_loss = portfolio_value * target_loss_pct

            breaking_scenarios: list[dict[str, Any]] = []

            for shock in np.arange(-0.05, -0.50, -0.05):
                scenario = StressScenario(
                    name=f"Reverse_test_{shock:.0%}",
                    scenario_type=ScenarioType.REVERSE,
                    equity_shock=shock,
                    volatility_shock=1.5,
                )

                result = await self.run_stress_test(scenario, portfolio, positions)

                if abs(result.portfolio_loss) >= target_loss:
                    breaking_scenarios.append({
                        "equity_shock": shock,
                        "loss": result.portfolio_loss,
                        "loss_pct": result.portfolio_loss_pct,
                    })
                    break

            for event, scenario in HISTORICAL_SCENARIOS.items():
                result = await self.run_stress_test(scenario, portfolio, positions)

                if abs(result.portfolio_loss) >= target_loss:
                    breaking_scenarios.append({
                        "event": event.value,
                        "scenario": scenario.name,
                        "loss": result.portfolio_loss,
                        "loss_pct": result.portfolio_loss_pct,
                    })

            return {
                "target_loss_pct": target_loss_pct,
                "target_loss_amount": target_loss,
                "breaking_scenarios": breaking_scenarios,
                "most_likely_breach": breaking_scenarios[0] if breaking_scenarios else None,
            }

        except Exception as e:
            logger.error(f"Reverse stress test failed: {e}")
            return {"error": str(e)}

    async def run_all_historical_scenarios(
        self,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> list[StressTestResult]:
        """
        Run all historical stress scenarios.

        Args:
            portfolio: Portfolio information
            positions: Current positions

        Returns:
            List of stress test results
        """
        results: list[StressTestResult] = []

        for event in HistoricalEvent:
            if event == HistoricalEvent.CUSTOM:
                continue

            try:
                result = await self.run_historical_stress_test(
                    event, portfolio, positions
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run {event.value} scenario: {e}")

        return results

    def add_custom_scenario(self, scenario: StressScenario) -> None:
        """
        Add a custom stress scenario.

        Args:
            scenario: Custom scenario to add
        """
        self._custom_scenarios[scenario.name] = scenario
        logger.info(f"Added custom scenario: {scenario.name}")

    def remove_custom_scenario(self, name: str) -> bool:
        """
        Remove a custom scenario.

        Args:
            name: Scenario name

        Returns:
            Whether removal was successful
        """
        if name in self._custom_scenarios:
            del self._custom_scenarios[name]
            return True
        return False

    def get_scenario(
        self,
        name: str | None = None,
        event: HistoricalEvent | None = None,
    ) -> StressScenario | None:
        """
        Get a stress scenario by name or event.

        Args:
            name: Scenario name
            event: Historical event

        Returns:
            StressScenario if found
        """
        if event:
            return HISTORICAL_SCENARIOS.get(event)
        if name:
            if name in self._custom_scenarios:
                return self._custom_scenarios[name]
            for scenario in HISTORICAL_SCENARIOS.values():
                if scenario.name == name:
                    return scenario
        return None

    def get_test_history(
        self,
        limit: int = 100,
    ) -> list[StressTestResult]:
        """Get recent test history."""
        return self._test_history[-limit:]

    async def get_stress_summary(
        self,
        portfolio: dict[str, Any],
        positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Get comprehensive stress test summary.

        Args:
            portfolio: Portfolio information
            positions: Current positions

        Returns:
            Summary dictionary
        """
        historical_results = await self.run_all_historical_scenarios(
            portfolio, positions
        )

        worst_case = min(historical_results, key=lambda r: r.portfolio_loss_pct)
        best_case = max(historical_results, key=lambda r: r.portfolio_loss_pct)

        failures = [r for r in historical_results if not r.passed]

        avg_loss = np.mean([r.portfolio_loss_pct for r in historical_results])

        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio.get("equity", 0),
            "scenarios_tested": len(historical_results),
            "scenarios_passed": len(historical_results) - len(failures),
            "scenarios_failed": len(failures),
            "worst_case": {
                "scenario": worst_case.scenario.name,
                "loss_pct": worst_case.portfolio_loss_pct,
                "loss_amount": worst_case.portfolio_loss,
            },
            "best_case": {
                "scenario": best_case.scenario.name,
                "loss_pct": best_case.portfolio_loss_pct,
            },
            "average_loss_pct": avg_loss,
            "failed_scenarios": [
                {
                    "scenario": r.scenario.name,
                    "breach_limits": r.breach_limits,
                }
                for r in failures
            ],
        }
