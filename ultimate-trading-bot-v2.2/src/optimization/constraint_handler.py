"""
Constraint Handler for Optimization.

This module provides constraint handling mechanisms for
constrained optimization problems in trading strategies.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of constraints."""

    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOUND = "bound"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"


class ViolationHandling(str, Enum):
    """Methods for handling constraint violations."""

    PENALTY = "penalty"
    REPAIR = "repair"
    REJECT = "reject"
    DEATH_PENALTY = "death_penalty"
    ADAPTIVE_PENALTY = "adaptive_penalty"
    FEASIBILITY_FIRST = "feasibility_first"


class ConstraintConfig(BaseModel):
    """Configuration for constraint handling."""

    handling_method: ViolationHandling = Field(
        default=ViolationHandling.PENALTY,
        description="Violation handling method",
    )
    penalty_factor: float = Field(default=1000.0, description="Base penalty factor")
    adaptive_penalty: bool = Field(default=True, description="Adapt penalty dynamically")
    feasibility_tolerance: float = Field(default=1e-6, description="Feasibility tolerance")
    repair_iterations: int = Field(default=10, description="Max repair iterations")
    constraint_epsilon: float = Field(default=1e-8, description="Numerical tolerance")


@dataclass
class Constraint:
    """Single constraint definition."""

    name: str
    constraint_type: ConstraintType
    func: Callable[[dict[str, Any]], float]
    bounds: tuple[float, float] | None = None
    priority: int = 1
    is_active: bool = True
    tolerance: float = 1e-6


@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""

    constraint_name: str
    violation_amount: float
    constraint_type: ConstraintType
    is_satisfied: bool
    penalty: float = 0.0


@dataclass
class FeasibilityResult:
    """Result of feasibility check."""

    is_feasible: bool
    violations: list[ConstraintViolation]
    total_violation: float
    total_penalty: float
    most_violated: str | None = None


class ConstraintHandler:
    """Handler for optimization constraints."""

    def __init__(
        self,
        config: ConstraintConfig | None = None,
    ) -> None:
        """
        Initialize constraint handler.

        Args:
            config: Constraint handling configuration
        """
        self.config = config or ConstraintConfig()
        self.constraints: list[Constraint] = []
        self._penalty_history: list[float] = []
        self._feasible_count = 0
        self._infeasible_count = 0

        logger.info(
            f"ConstraintHandler initialized with {self.config.handling_method.value} method"
        )

    def add_constraint(
        self,
        name: str,
        func: Callable[[dict[str, Any]], float],
        constraint_type: ConstraintType = ConstraintType.INEQUALITY,
        bounds: tuple[float, float] | None = None,
        priority: int = 1,
    ) -> None:
        """
        Add a constraint.

        Args:
            name: Constraint name
            func: Constraint function (returns value, <=0 for inequality)
            constraint_type: Type of constraint
            bounds: Optional bounds for equality constraints
            priority: Constraint priority (higher = more important)
        """
        constraint = Constraint(
            name=name,
            constraint_type=constraint_type,
            func=func,
            bounds=bounds,
            priority=priority,
        )
        self.constraints.append(constraint)
        logger.debug(f"Added constraint: {name}")

    def add_bound_constraint(
        self,
        param_name: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> None:
        """
        Add parameter bound constraint.

        Args:
            param_name: Parameter name
            lower: Lower bound
            upper: Upper bound
        """
        if lower is not None:
            def lower_func(params: dict, pn: str = param_name, lb: float = lower) -> float:
                return lb - params.get(pn, lb)

            self.add_constraint(
                name=f"{param_name}_lower_bound",
                func=lower_func,
                constraint_type=ConstraintType.BOUND,
            )

        if upper is not None:
            def upper_func(params: dict, pn: str = param_name, ub: float = upper) -> float:
                return params.get(pn, ub) - ub

            self.add_constraint(
                name=f"{param_name}_upper_bound",
                func=upper_func,
                constraint_type=ConstraintType.BOUND,
            )

    def add_linear_constraint(
        self,
        coefficients: dict[str, float],
        bound: float,
        is_equality: bool = False,
    ) -> None:
        """
        Add linear constraint: sum(coef * param) <= bound.

        Args:
            coefficients: Parameter coefficients
            bound: Constraint bound
            is_equality: True for equality constraint
        """
        def linear_func(params: dict) -> float:
            total = sum(
                coef * params.get(name, 0)
                for name, coef in coefficients.items()
            )
            return total - bound

        ctype = ConstraintType.EQUALITY if is_equality else ConstraintType.LINEAR
        name = f"linear_{'eq' if is_equality else 'ineq'}_{len(self.constraints)}"

        self.add_constraint(
            name=name,
            func=linear_func,
            constraint_type=ctype,
        )

    def add_max_drawdown_constraint(
        self,
        max_drawdown: float,
        returns_func: Callable[[dict[str, Any]], np.ndarray],
    ) -> None:
        """
        Add maximum drawdown constraint.

        Args:
            max_drawdown: Maximum allowed drawdown (negative value)
            returns_func: Function to get returns from parameters
        """
        def dd_constraint(params: dict) -> float:
            returns = returns_func(params)
            cum_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            actual_dd = np.min(drawdowns)
            return max_drawdown - actual_dd

        self.add_constraint(
            name="max_drawdown",
            func=dd_constraint,
            constraint_type=ConstraintType.INEQUALITY,
            priority=2,
        )

    def add_min_sharpe_constraint(
        self,
        min_sharpe: float,
        returns_func: Callable[[dict[str, Any]], np.ndarray],
        annualization: float = 252.0,
    ) -> None:
        """
        Add minimum Sharpe ratio constraint.

        Args:
            min_sharpe: Minimum required Sharpe ratio
            returns_func: Function to get returns from parameters
            annualization: Annualization factor
        """
        def sharpe_constraint(params: dict) -> float:
            returns = returns_func(params)
            if len(returns) < 2 or np.std(returns) == 0:
                return min_sharpe
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(annualization)
            return min_sharpe - sharpe

        self.add_constraint(
            name="min_sharpe",
            func=sharpe_constraint,
            constraint_type=ConstraintType.INEQUALITY,
            priority=2,
        )

    def add_position_limit_constraint(
        self,
        max_positions: int,
        position_func: Callable[[dict[str, Any]], int],
    ) -> None:
        """
        Add maximum position constraint.

        Args:
            max_positions: Maximum number of positions
            position_func: Function to get position count
        """
        def position_constraint(params: dict) -> float:
            n_positions = position_func(params)
            return n_positions - max_positions

        self.add_constraint(
            name="max_positions",
            func=position_constraint,
            constraint_type=ConstraintType.INEQUALITY,
        )

    def check_feasibility(
        self,
        params: dict[str, Any],
    ) -> FeasibilityResult:
        """
        Check if parameters satisfy all constraints.

        Args:
            params: Parameter dictionary

        Returns:
            Feasibility result
        """
        violations = []
        total_violation = 0.0
        total_penalty = 0.0
        max_violation = 0.0
        most_violated = None

        for constraint in self.constraints:
            if not constraint.is_active:
                continue

            try:
                value = constraint.func(params)
            except Exception as e:
                logger.warning(f"Constraint {constraint.name} evaluation failed: {e}")
                value = float("inf")

            is_satisfied = True
            violation_amount = 0.0

            if constraint.constraint_type == ConstraintType.EQUALITY:
                if constraint.bounds:
                    if not (constraint.bounds[0] - constraint.tolerance <= value <= constraint.bounds[1] + constraint.tolerance):
                        is_satisfied = False
                        violation_amount = min(
                            abs(value - constraint.bounds[0]),
                            abs(value - constraint.bounds[1]),
                        )
                else:
                    if abs(value) > constraint.tolerance:
                        is_satisfied = False
                        violation_amount = abs(value)

            elif constraint.constraint_type in [
                ConstraintType.INEQUALITY,
                ConstraintType.BOUND,
                ConstraintType.LINEAR,
                ConstraintType.NONLINEAR,
            ]:
                if value > constraint.tolerance:
                    is_satisfied = False
                    violation_amount = value

            penalty = 0.0
            if not is_satisfied:
                penalty = self._calculate_penalty(violation_amount, constraint.priority)
                total_violation += violation_amount
                total_penalty += penalty

                if violation_amount > max_violation:
                    max_violation = violation_amount
                    most_violated = constraint.name

            violations.append(ConstraintViolation(
                constraint_name=constraint.name,
                violation_amount=violation_amount,
                constraint_type=constraint.constraint_type,
                is_satisfied=is_satisfied,
                penalty=penalty,
            ))

        is_feasible = all(v.is_satisfied for v in violations)

        if is_feasible:
            self._feasible_count += 1
        else:
            self._infeasible_count += 1

        return FeasibilityResult(
            is_feasible=is_feasible,
            violations=violations,
            total_violation=total_violation,
            total_penalty=total_penalty,
            most_violated=most_violated,
        )

    def _calculate_penalty(
        self,
        violation: float,
        priority: int,
    ) -> float:
        """Calculate penalty for constraint violation."""
        base_penalty = self.config.penalty_factor

        if self.config.adaptive_penalty:
            if self._feasible_count + self._infeasible_count > 0:
                feasible_ratio = self._feasible_count / (self._feasible_count + self._infeasible_count)
                if feasible_ratio < 0.3:
                    base_penalty *= 0.5
                elif feasible_ratio > 0.7:
                    base_penalty *= 2.0

        return base_penalty * priority * violation ** 2

    def handle_violation(
        self,
        params: dict[str, Any],
        feasibility: FeasibilityResult,
    ) -> tuple[dict[str, Any], float]:
        """
        Handle constraint violations according to configured method.

        Args:
            params: Original parameters
            feasibility: Feasibility check result

        Returns:
            Tuple of (possibly repaired params, penalty)
        """
        if feasibility.is_feasible:
            return params, 0.0

        method = self.config.handling_method

        if method == ViolationHandling.PENALTY:
            return params, feasibility.total_penalty

        elif method == ViolationHandling.ADAPTIVE_PENALTY:
            return params, self._adaptive_penalty(feasibility)

        elif method == ViolationHandling.REPAIR:
            repaired = self._repair_solution(params, feasibility)
            return repaired, 0.0

        elif method == ViolationHandling.REJECT:
            return params, float("inf")

        elif method == ViolationHandling.DEATH_PENALTY:
            return params, float("inf")

        elif method == ViolationHandling.FEASIBILITY_FIRST:
            return params, feasibility.total_penalty

        return params, feasibility.total_penalty

    def _adaptive_penalty(
        self,
        feasibility: FeasibilityResult,
    ) -> float:
        """Calculate adaptive penalty."""
        total = self._feasible_count + self._infeasible_count
        if total == 0:
            return feasibility.total_penalty

        feasible_ratio = self._feasible_count / total

        if feasible_ratio < 0.2:
            multiplier = 0.5
        elif feasible_ratio < 0.4:
            multiplier = 0.8
        elif feasible_ratio > 0.8:
            multiplier = 2.0
        elif feasible_ratio > 0.6:
            multiplier = 1.5
        else:
            multiplier = 1.0

        return feasibility.total_penalty * multiplier

    def _repair_solution(
        self,
        params: dict[str, Any],
        feasibility: FeasibilityResult,
    ) -> dict[str, Any]:
        """
        Attempt to repair infeasible solution.

        Args:
            params: Infeasible parameters
            feasibility: Feasibility result

        Returns:
            Repaired parameters
        """
        repaired = params.copy()

        for _ in range(self.config.repair_iterations):
            new_feasibility = self.check_feasibility(repaired)
            if new_feasibility.is_feasible:
                break

            for violation in new_feasibility.violations:
                if violation.is_satisfied:
                    continue

                constraint = next(
                    (c for c in self.constraints if c.name == violation.constraint_name),
                    None,
                )

                if constraint and constraint.constraint_type == ConstraintType.BOUND:
                    repaired = self._repair_bound_violation(
                        repaired, constraint, violation
                    )

        return repaired

    def _repair_bound_violation(
        self,
        params: dict[str, Any],
        constraint: Constraint,
        violation: ConstraintViolation,
    ) -> dict[str, Any]:
        """Repair bound constraint violation."""
        repaired = params.copy()

        parts = constraint.name.split("_")
        if len(parts) >= 3:
            param_name = "_".join(parts[:-2])
            bound_type = parts[-2]

            if param_name in repaired:
                current = repaired[param_name]
                if bound_type == "lower":
                    repaired[param_name] = current + violation.violation_amount * 1.1
                elif bound_type == "upper":
                    repaired[param_name] = current - violation.violation_amount * 1.1

        return repaired

    def get_feasibility_stats(self) -> dict[str, Any]:
        """Get constraint feasibility statistics."""
        total = self._feasible_count + self._infeasible_count
        return {
            "total_evaluations": total,
            "feasible_count": self._feasible_count,
            "infeasible_count": self._infeasible_count,
            "feasibility_ratio": self._feasible_count / total if total > 0 else 0.0,
            "num_constraints": len(self.constraints),
            "active_constraints": sum(1 for c in self.constraints if c.is_active),
        }

    def reset_stats(self) -> None:
        """Reset feasibility statistics."""
        self._feasible_count = 0
        self._infeasible_count = 0
        self._penalty_history.clear()


class ConstraintSetBuilder:
    """Builder for creating constraint sets."""

    def __init__(self) -> None:
        """Initialize constraint set builder."""
        self.handler = ConstraintHandler()

    def with_bounds(
        self,
        param_name: str,
        lower: float | None = None,
        upper: float | None = None,
    ) -> "ConstraintSetBuilder":
        """Add bound constraints."""
        self.handler.add_bound_constraint(param_name, lower, upper)
        return self

    def with_linear(
        self,
        coefficients: dict[str, float],
        bound: float,
        is_equality: bool = False,
    ) -> "ConstraintSetBuilder":
        """Add linear constraint."""
        self.handler.add_linear_constraint(coefficients, bound, is_equality)
        return self

    def with_max_drawdown(
        self,
        max_dd: float,
        returns_func: Callable[[dict[str, Any]], np.ndarray],
    ) -> "ConstraintSetBuilder":
        """Add max drawdown constraint."""
        self.handler.add_max_drawdown_constraint(max_dd, returns_func)
        return self

    def with_min_sharpe(
        self,
        min_sharpe: float,
        returns_func: Callable[[dict[str, Any]], np.ndarray],
    ) -> "ConstraintSetBuilder":
        """Add min Sharpe constraint."""
        self.handler.add_min_sharpe_constraint(min_sharpe, returns_func)
        return self

    def with_custom(
        self,
        name: str,
        func: Callable[[dict[str, Any]], float],
        priority: int = 1,
    ) -> "ConstraintSetBuilder":
        """Add custom constraint."""
        self.handler.add_constraint(name, func, priority=priority)
        return self

    def with_config(
        self,
        config: ConstraintConfig,
    ) -> "ConstraintSetBuilder":
        """Set constraint handler config."""
        self.handler.config = config
        return self

    def build(self) -> ConstraintHandler:
        """Build and return constraint handler."""
        return self.handler


def create_constraint_handler(
    handling_method: str = "penalty",
    penalty_factor: float = 1000.0,
    adaptive: bool = True,
) -> ConstraintHandler:
    """
    Create a constraint handler.

    Args:
        handling_method: Violation handling method
        penalty_factor: Base penalty factor
        adaptive: Use adaptive penalty

    Returns:
        Configured ConstraintHandler
    """
    config = ConstraintConfig(
        handling_method=ViolationHandling(handling_method),
        penalty_factor=penalty_factor,
        adaptive_penalty=adaptive,
    )
    return ConstraintHandler(config)
