"""
Optimization Package for Ultimate Trading Bot v2.2.

This package provides comprehensive optimization algorithms for
trading strategy parameter tuning, including genetic algorithms,
Bayesian optimization, particle swarm, and differential evolution.
"""

from ultimate_trading_bot.optimization.base_optimizer import (
    BaseOptimizer,
    BaseOptimizerConfig,
    CategoricalParameter,
    EvaluationCache,
    EvaluationPoint,
    GridSearchOptimizer,
    OptimizationResult,
    OptimizationStatus,
    OptimizationType,
    ParameterBounds,
    RandomSearchOptimizer,
    SearchSpace,
    create_grid_optimizer,
    create_random_optimizer,
)
from ultimate_trading_bot.optimization.bayesian_optimizer import (
    AcquisitionFunction,
    BayesianOptimizer,
    BayesianOptimizerConfig,
    GaussianProcessModel,
    KernelType,
    create_bayesian_optimizer,
)
from ultimate_trading_bot.optimization.constraint_handler import (
    Constraint,
    ConstraintConfig,
    ConstraintHandler,
    ConstraintSetBuilder,
    ConstraintType,
    ConstraintViolation,
    FeasibilityResult,
    ViolationHandling,
    create_constraint_handler,
)
from ultimate_trading_bot.optimization.differential_evolution import (
    AdaptiveDE,
    CrossoverType,
    DEConfig,
    DEIndividual,
    DEPopulation,
    DifferentialEvolution,
    MutationStrategy,
    SHADE,
    create_de_optimizer,
)
from ultimate_trading_bot.optimization.genetic_optimizer import (
    CrossoverMethod,
    GeneticOptimizer,
    GeneticOptimizerConfig,
    Individual,
    MutationMethod,
    NSGA2Optimizer,
    Population,
    SelectionMethod,
    create_genetic_optimizer,
)
from ultimate_trading_bot.optimization.hyperparameter_tuner import (
    CrossValidation,
    CrossValidator,
    CVFold,
    CVResult,
    HyperparameterTuner,
    SuccessiveHalvingTuner,
    TunerConfig,
    TuningMethod,
    TuningResult,
    create_hyperparameter_tuner,
)
from ultimate_trading_bot.optimization.objective_functions import (
    MultiObjectiveFunction,
    ObjectiveFunction,
    ObjectiveFunctionConfig,
    ObjectiveResult,
    ObjectiveType,
    PerformanceMetrics,
    RiskMeasure,
    create_objective_function,
)
from ultimate_trading_bot.optimization.particle_swarm import (
    AdaptivePSO,
    CompetitivePSO,
    Particle,
    ParticleSwarmOptimizer,
    PSOConfig,
    Swarm,
    TopologyType,
    VelocityUpdate,
    create_pso_optimizer,
)

__all__ = [
    # Base Optimizer
    "BaseOptimizer",
    "BaseOptimizerConfig",
    "CategoricalParameter",
    "EvaluationCache",
    "EvaluationPoint",
    "GridSearchOptimizer",
    "OptimizationResult",
    "OptimizationStatus",
    "OptimizationType",
    "ParameterBounds",
    "RandomSearchOptimizer",
    "SearchSpace",
    "create_grid_optimizer",
    "create_random_optimizer",
    # Bayesian Optimizer
    "AcquisitionFunction",
    "BayesianOptimizer",
    "BayesianOptimizerConfig",
    "GaussianProcessModel",
    "KernelType",
    "create_bayesian_optimizer",
    # Constraint Handler
    "Constraint",
    "ConstraintConfig",
    "ConstraintHandler",
    "ConstraintSetBuilder",
    "ConstraintType",
    "ConstraintViolation",
    "FeasibilityResult",
    "ViolationHandling",
    "create_constraint_handler",
    # Differential Evolution
    "AdaptiveDE",
    "CrossoverType",
    "DEConfig",
    "DEIndividual",
    "DEPopulation",
    "DifferentialEvolution",
    "MutationStrategy",
    "SHADE",
    "create_de_optimizer",
    # Genetic Optimizer
    "CrossoverMethod",
    "GeneticOptimizer",
    "GeneticOptimizerConfig",
    "Individual",
    "MutationMethod",
    "NSGA2Optimizer",
    "Population",
    "SelectionMethod",
    "create_genetic_optimizer",
    # Hyperparameter Tuner
    "CrossValidation",
    "CrossValidator",
    "CVFold",
    "CVResult",
    "HyperparameterTuner",
    "SuccessiveHalvingTuner",
    "TunerConfig",
    "TuningMethod",
    "TuningResult",
    "create_hyperparameter_tuner",
    # Objective Functions
    "MultiObjectiveFunction",
    "ObjectiveFunction",
    "ObjectiveFunctionConfig",
    "ObjectiveResult",
    "ObjectiveType",
    "PerformanceMetrics",
    "RiskMeasure",
    "create_objective_function",
    # Particle Swarm
    "AdaptivePSO",
    "CompetitivePSO",
    "Particle",
    "ParticleSwarmOptimizer",
    "PSOConfig",
    "Swarm",
    "TopologyType",
    "VelocityUpdate",
    "create_pso_optimizer",
]


def create_optimizer(
    method: str = "bayesian",
    max_iterations: int = 100,
    optimization_type: str = "maximization",
    seed: int | None = None,
    **kwargs,
) -> BaseOptimizer:
    """
    Create an optimizer based on method name.

    Args:
        method: Optimization method
        max_iterations: Maximum iterations
        optimization_type: Optimization type
        seed: Random seed
        **kwargs: Method-specific arguments

    Returns:
        Configured optimizer
    """
    method = method.lower()

    if method == "bayesian":
        return create_bayesian_optimizer(
            max_iterations=max_iterations,
            optimization_type=optimization_type,
            seed=seed,
            **kwargs,
        )
    elif method == "genetic":
        return create_genetic_optimizer(
            max_iterations=max_iterations,
            optimization_type=optimization_type,
            seed=seed,
            **kwargs,
        )
    elif method == "pso":
        return create_pso_optimizer(
            max_iterations=max_iterations,
            optimization_type=optimization_type,
            seed=seed,
            **kwargs,
        )
    elif method == "de":
        return create_de_optimizer(
            max_iterations=max_iterations,
            optimization_type=optimization_type,
            seed=seed,
            **kwargs,
        )
    elif method == "random":
        return create_random_optimizer(
            max_iterations=max_iterations,
            optimization_type=optimization_type,
            seed=seed,
            **kwargs,
        )
    elif method == "grid":
        return create_grid_optimizer(
            optimization_type=optimization_type,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def create_search_space() -> SearchSpace:
    """
    Create an empty search space for configuration.

    Returns:
        Empty SearchSpace
    """
    return SearchSpace()


def quick_optimize(
    objective_func,
    parameter_ranges: dict[str, tuple[float, float]],
    method: str = "bayesian",
    max_iterations: int = 100,
    is_integer: dict[str, bool] | None = None,
) -> OptimizationResult:
    """
    Quick optimization helper function.

    Args:
        objective_func: Objective function to optimize
        parameter_ranges: Dictionary of parameter name -> (min, max)
        method: Optimization method
        max_iterations: Maximum iterations
        is_integer: Dictionary indicating integer parameters

    Returns:
        Optimization result
    """
    import asyncio

    is_integer = is_integer or {}

    search_space = SearchSpace()
    for name, (lo, hi) in parameter_ranges.items():
        search_space.continuous_params.append(
            ParameterBounds(
                name=name,
                lower=lo,
                upper=hi,
                is_integer=is_integer.get(name, False),
            )
        )

    optimizer = create_optimizer(
        method=method,
        max_iterations=max_iterations,
    )

    return asyncio.run(optimizer.optimize(objective_func, search_space))
