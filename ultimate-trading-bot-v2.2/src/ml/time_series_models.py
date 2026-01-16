"""
Time Series Models Module for Ultimate Trading Bot v2.2

Implements ARIMA, GARCH, Exponential Smoothing, VAR, and other time series
forecasting models specifically designed for financial time series.

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


class TimeSeriesModelType(Enum):
    """Types of time series models."""
    ARIMA = "arima"
    SARIMA = "sarima"
    GARCH = "garch"
    EGARCH = "egarch"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    HOLT_WINTERS = "holt_winters"
    VAR = "var"
    VECM = "vecm"
    THETA = "theta"


class SeasonalityType(Enum):
    """Types of seasonality."""
    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class TrendType(Enum):
    """Types of trend."""
    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    DAMPED = "damped"


@dataclass
class ARIMAOrder:
    """ARIMA model order (p, d, q)."""
    p: int
    d: int
    q: int

    def to_tuple(self) -> tuple[int, int, int]:
        """Convert to tuple."""
        return (self.p, self.d, self.q)


@dataclass
class SARIMAOrder:
    """SARIMA seasonal order (P, D, Q, s)."""
    P: int
    D: int
    Q: int
    s: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Convert to tuple."""
        return (self.P, self.D, self.Q, self.s)


@dataclass
class GARCHOrder:
    """GARCH model order (p, q)."""
    p: int
    q: int

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.p, self.q)


@dataclass
class TimeSeriesForecast:
    """Time series forecast result."""
    forecast: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    confidence_level: float
    timestamps: Optional[np.ndarray] = None
    model_name: str = ""
    forecast_horizon: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "forecast": self.forecast.tolist(),
            "confidence_lower": self.confidence_lower.tolist(),
            "confidence_upper": self.confidence_upper.tolist(),
            "confidence_level": self.confidence_level,
            "model_name": self.model_name,
            "forecast_horizon": self.forecast_horizon
        }


@dataclass
class ModelDiagnostics:
    """Model diagnostic statistics."""
    aic: float
    bic: float
    log_likelihood: float
    residual_mean: float
    residual_std: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    arch_lm_stat: float
    arch_lm_pvalue: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "aic": self.aic,
            "bic": self.bic,
            "log_likelihood": self.log_likelihood,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "ljung_box_stat": self.ljung_box_stat,
            "ljung_box_pvalue": self.ljung_box_pvalue,
            "jarque_bera_stat": self.jarque_bera_stat,
            "jarque_bera_pvalue": self.jarque_bera_pvalue,
            "arch_lm_stat": self.arch_lm_stat,
            "arch_lm_pvalue": self.arch_lm_pvalue
        }


@dataclass
class FittedModel:
    """Container for fitted model."""
    model_type: TimeSeriesModelType
    parameters: dict[str, Any]
    residuals: np.ndarray
    fitted_values: np.ndarray
    diagnostics: ModelDiagnostics
    training_data: np.ndarray
    fit_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "parameters": self.parameters,
            "diagnostics": self.diagnostics.to_dict(),
            "fit_timestamp": self.fit_timestamp.isoformat()
        }


class BaseTimeSeriesModel(ABC):
    """Base class for time series models."""

    def __init__(self, name: str):
        """
        Initialize model.

        Args:
            name: Model name
        """
        self.name = name
        self._fitted: Optional[FittedModel] = None
        self._is_fitted = False

        logger.info(f"Initialized {self.__class__.__name__}: {name}")

    @abstractmethod
    async def fit(self, data: np.ndarray, **kwargs: Any) -> FittedModel:
        """Fit the model to data."""
        pass

    @abstractmethod
    async def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """Generate forecasts."""
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    def _calculate_diagnostics(
        self,
        residuals: np.ndarray,
        n_params: int
    ) -> ModelDiagnostics:
        """
        Calculate model diagnostics.

        Args:
            residuals: Model residuals
            n_params: Number of parameters

        Returns:
            ModelDiagnostics object
        """
        n = len(residuals)

        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        sse = np.sum(residuals ** 2)
        log_likelihood = -n / 2 * (1 + np.log(2 * np.pi) + np.log(sse / n))

        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood

        ljung_box_stat, ljung_box_pvalue = self._ljung_box_test(residuals)

        jb_stat, jb_pvalue = self._jarque_bera_test(residuals)

        arch_stat, arch_pvalue = self._arch_lm_test(residuals)

        return ModelDiagnostics(
            aic=float(aic),
            bic=float(bic),
            log_likelihood=float(log_likelihood),
            residual_mean=residual_mean,
            residual_std=residual_std,
            ljung_box_stat=ljung_box_stat,
            ljung_box_pvalue=ljung_box_pvalue,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            arch_lm_stat=arch_stat,
            arch_lm_pvalue=arch_pvalue
        )

    def _ljung_box_test(
        self,
        residuals: np.ndarray,
        lags: int = 10
    ) -> tuple[float, float]:
        """Perform Ljung-Box test for autocorrelation."""
        n = len(residuals)
        acf_values = self._calculate_acf(residuals, lags)

        q_stat = n * (n + 2) * np.sum(acf_values[1:] ** 2 / (n - np.arange(1, lags + 1)))

        p_value = 1 - self._chi2_cdf(q_stat, lags)

        return float(q_stat), float(p_value)

    def _jarque_bera_test(self, residuals: np.ndarray) -> tuple[float, float]:
        """Perform Jarque-Bera test for normality."""
        n = len(residuals)

        mean = np.mean(residuals)
        std = np.std(residuals)

        if std == 0:
            return 0.0, 1.0

        standardized = (residuals - mean) / std

        skewness = np.mean(standardized ** 3)
        kurtosis = np.mean(standardized ** 4) - 3

        jb_stat = n / 6 * (skewness ** 2 + kurtosis ** 2 / 4)

        p_value = 1 - self._chi2_cdf(jb_stat, 2)

        return float(jb_stat), float(p_value)

    def _arch_lm_test(
        self,
        residuals: np.ndarray,
        lags: int = 5
    ) -> tuple[float, float]:
        """Perform ARCH LM test for heteroskedasticity."""
        n = len(residuals)
        squared_residuals = residuals ** 2

        if n <= lags + 1:
            return 0.0, 1.0

        y = squared_residuals[lags:]
        X = np.column_stack([
            squared_residuals[lags - i - 1:n - i - 1]
            for i in range(lags)
        ])

        X = np.column_stack([np.ones(len(y)), X])

        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            lm_stat = (n - lags) * r_squared
            p_value = 1 - self._chi2_cdf(lm_stat, lags)

            return float(lm_stat), float(p_value)

        except Exception:
            return 0.0, 1.0

    def _calculate_acf(
        self,
        data: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """Calculate autocorrelation function."""
        n = len(data)
        mean = np.mean(data)
        var = np.var(data)

        if var == 0:
            return np.zeros(max_lag + 1)

        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for lag in range(1, max_lag + 1):
            acf[lag] = np.sum(
                (data[lag:] - mean) * (data[:-lag] - mean)
            ) / (n * var)

        return acf

    def _chi2_cdf(self, x: float, df: int) -> float:
        """Chi-squared CDF (approximation)."""
        if x <= 0:
            return 0.0

        k = df / 2

        gamma_k = self._gamma_function(k)

        result = self._lower_incomplete_gamma(k, x / 2) / gamma_k

        return min(max(result, 0.0), 1.0)

    def _gamma_function(self, z: float) -> float:
        """Gamma function using Lanczos approximation."""
        if z < 0.5:
            return np.pi / (np.sin(np.pi * z) * self._gamma_function(1 - z))

        z -= 1
        g = 7
        c = [
            0.99999999999980993,
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ]

        x = c[0]
        for i in range(1, g + 2):
            x += c[i] / (z + i)

        t = z + g + 0.5
        return np.sqrt(2 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x

    def _lower_incomplete_gamma(self, a: float, x: float) -> float:
        """Lower incomplete gamma function."""
        if x <= 0:
            return 0.0

        if x < a + 1:
            term = 1.0 / a
            sum_val = term

            for n in range(1, 100):
                term *= x / (a + n)
                sum_val += term

                if abs(term) < 1e-10:
                    break

            return sum_val * np.exp(-x + a * np.log(x))
        else:
            return self._gamma_function(a) - self._upper_incomplete_gamma(a, x)

    def _upper_incomplete_gamma(self, a: float, x: float) -> float:
        """Upper incomplete gamma function using continued fraction."""
        b = x + 1 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d

        for i in range(1, 100):
            an = -i * (i - a)
            b += 2
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta

            if abs(delta - 1) < 1e-10:
                break

        return np.exp(-x + a * np.log(x)) * h


class ARIMAModel(BaseTimeSeriesModel):
    """ARIMA (AutoRegressive Integrated Moving Average) model."""

    def __init__(
        self,
        order: ARIMAOrder,
        name: str = "ARIMA"
    ):
        """
        Initialize ARIMA model.

        Args:
            order: ARIMA order (p, d, q)
            name: Model name
        """
        super().__init__(name)
        self.order = order
        self._ar_params: Optional[np.ndarray] = None
        self._ma_params: Optional[np.ndarray] = None
        self._constant: float = 0.0
        self._sigma2: float = 1.0
        self._original_data: Optional[np.ndarray] = None
        self._differenced_data: Optional[np.ndarray] = None

    async def fit(
        self,
        data: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs: Any
    ) -> FittedModel:
        """
        Fit ARIMA model.

        Args:
            data: Time series data
            max_iter: Maximum iterations
            tol: Convergence tolerance
            **kwargs: Additional arguments

        Returns:
            FittedModel object
        """
        try:
            logger.info(f"Fitting ARIMA{self.order.to_tuple()} model")

            self._original_data = data.copy()

            differenced = self._difference(data, self.order.d)
            self._differenced_data = differenced

            self._ar_params, self._ma_params, self._constant, self._sigma2 = \
                self._estimate_parameters(differenced, max_iter, tol)

            fitted_diff = self._compute_fitted(differenced)
            residuals = differenced[max(self.order.p, self.order.q):] - fitted_diff

            fitted_values = self._integrate_fitted(fitted_diff, data, self.order.d)

            n_params = self.order.p + self.order.q + 2
            diagnostics = self._calculate_diagnostics(residuals, n_params)

            self._fitted = FittedModel(
                model_type=TimeSeriesModelType.ARIMA,
                parameters={
                    "order": self.order.to_tuple(),
                    "ar_params": self._ar_params.tolist() if self._ar_params is not None else [],
                    "ma_params": self._ma_params.tolist() if self._ma_params is not None else [],
                    "constant": self._constant,
                    "sigma2": self._sigma2
                },
                residuals=residuals,
                fitted_values=fitted_values,
                diagnostics=diagnostics,
                training_data=data
            )

            self._is_fitted = True

            logger.info(
                f"ARIMA fit complete: AIC={diagnostics.aic:.2f}, "
                f"BIC={diagnostics.bic:.2f}"
            )

            return self._fitted

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise

    async def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """
        Generate ARIMA forecasts.

        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level for intervals

        Returns:
            TimeSeriesForecast object
        """
        if not self._is_fitted or self._fitted is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            logger.info(f"Generating {horizon}-step ARIMA forecast")

            forecasts_diff = self._forecast_differenced(horizon)

            forecasts = self._integrate_forecasts(
                forecasts_diff,
                self._original_data,
                self.order.d
            )

            z_alpha = self._ppf_normal((1 + confidence_level) / 2)

            forecast_var = self._compute_forecast_variance(horizon)
            se = np.sqrt(forecast_var)

            lower = forecasts - z_alpha * se
            upper = forecasts + z_alpha * se

            result = TimeSeriesForecast(
                forecast=forecasts,
                confidence_lower=lower,
                confidence_upper=upper,
                confidence_level=confidence_level,
                model_name=self.name,
                forecast_horizon=horizon
            )

            logger.info(f"Forecast generated: {forecasts[:min(3, horizon)]}...")

            return result

        except Exception as e:
            logger.error(f"Error generating ARIMA forecast: {e}")
            raise

    def _difference(self, data: np.ndarray, d: int) -> np.ndarray:
        """Apply differencing."""
        result = data.copy()
        for _ in range(d):
            result = np.diff(result)
        return result

    def _estimate_parameters(
        self,
        data: np.ndarray,
        max_iter: int,
        tol: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Estimate ARIMA parameters using OLS/CSS method."""
        p, q = self.order.p, self.order.q
        n = len(data)

        constant = np.mean(data)
        centered_data = data - constant

        if p > 0:
            ar_params = self._estimate_ar_params(centered_data, p)
        else:
            ar_params = np.array([])

        if q > 0:
            if p > 0:
                ar_residuals = centered_data[p:] - self._apply_ar(centered_data, ar_params)
            else:
                ar_residuals = centered_data

            ma_params = self._estimate_ma_params(ar_residuals, q)
        else:
            ma_params = np.array([])

        for _ in range(max_iter):
            residuals = self._compute_residuals(centered_data, ar_params, ma_params)
            sigma2 = np.var(residuals)

            new_ar = self._update_ar_params(centered_data, residuals, ar_params, ma_params)
            new_ma = self._update_ma_params(centered_data, residuals, ar_params, ma_params)

            if (np.max(np.abs(new_ar - ar_params)) < tol and
                np.max(np.abs(new_ma - ma_params)) < tol):
                break

            ar_params = new_ar
            ma_params = new_ma

        residuals = self._compute_residuals(centered_data, ar_params, ma_params)
        sigma2 = float(np.var(residuals))

        return ar_params, ma_params, constant, sigma2

    def _estimate_ar_params(self, data: np.ndarray, p: int) -> np.ndarray:
        """Estimate AR parameters using Yule-Walker."""
        n = len(data)

        r = np.zeros(p + 1)
        for k in range(p + 1):
            r[k] = np.sum(data[:n - k] * data[k:]) / n

        R = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                R[i, j] = r[abs(i - j)]

        try:
            ar_params = np.linalg.solve(R, r[1:p + 1])
        except np.linalg.LinAlgError:
            ar_params = np.zeros(p)

        return ar_params

    def _estimate_ma_params(self, residuals: np.ndarray, q: int) -> np.ndarray:
        """Estimate MA parameters using innovation algorithm."""
        n = len(residuals)

        r = np.zeros(q + 1)
        for k in range(q + 1):
            if k < n:
                r[k] = np.sum(residuals[:n - k] * residuals[k:]) / n

        ma_params = np.zeros(q)
        for j in range(q):
            if r[0] != 0:
                ma_params[j] = r[j + 1] / r[0] * 0.5

        return ma_params

    def _apply_ar(self, data: np.ndarray, ar_params: np.ndarray) -> np.ndarray:
        """Apply AR component."""
        p = len(ar_params)
        n = len(data)

        result = np.zeros(n - p)
        for t in range(p, n):
            result[t - p] = np.sum(ar_params * data[t - p:t][::-1])

        return result

    def _compute_residuals(
        self,
        data: np.ndarray,
        ar_params: np.ndarray,
        ma_params: np.ndarray
    ) -> np.ndarray:
        """Compute model residuals."""
        p = len(ar_params)
        q = len(ma_params)
        n = len(data)
        start = max(p, q)

        residuals = np.zeros(n - start)

        for t in range(start, n):
            pred = 0.0

            if p > 0:
                pred += np.sum(ar_params * data[t - p:t][::-1])

            if q > 0:
                for j in range(min(q, t - start)):
                    pred += ma_params[j] * residuals[t - start - j - 1]

            residuals[t - start] = data[t] - pred

        return residuals

    def _update_ar_params(
        self,
        data: np.ndarray,
        residuals: np.ndarray,
        ar_params: np.ndarray,
        ma_params: np.ndarray
    ) -> np.ndarray:
        """Update AR parameters."""
        if len(ar_params) == 0:
            return ar_params

        p = len(ar_params)
        n = len(data)
        start = max(p, len(ma_params))

        X = np.zeros((n - start, p))
        for t in range(start, n):
            X[t - start] = data[t - p:t][::-1]

        y = data[start:] - self._apply_ma(residuals, ma_params, n - start)

        try:
            new_ar = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            new_ar = ar_params

        return new_ar

    def _update_ma_params(
        self,
        data: np.ndarray,
        residuals: np.ndarray,
        ar_params: np.ndarray,
        ma_params: np.ndarray
    ) -> np.ndarray:
        """Update MA parameters."""
        if len(ma_params) == 0:
            return ma_params

        new_residuals = self._compute_residuals(data, ar_params, ma_params)

        return self._estimate_ma_params(new_residuals, len(ma_params))

    def _apply_ma(
        self,
        residuals: np.ndarray,
        ma_params: np.ndarray,
        length: int
    ) -> np.ndarray:
        """Apply MA component."""
        q = len(ma_params)
        result = np.zeros(length)

        for t in range(length):
            for j in range(min(q, t)):
                if t - j - 1 < len(residuals):
                    result[t] += ma_params[j] * residuals[t - j - 1]

        return result

    def _compute_fitted(self, data: np.ndarray) -> np.ndarray:
        """Compute fitted values for differenced data."""
        start = max(self.order.p, self.order.q)
        n = len(data)

        fitted = np.zeros(n - start)
        residuals = np.zeros(n - start)

        for t in range(start, n):
            pred = self._constant

            if self.order.p > 0 and self._ar_params is not None:
                pred += np.sum(self._ar_params * (data[t - self.order.p:t] - self._constant)[::-1])

            if self.order.q > 0 and self._ma_params is not None:
                for j in range(min(self.order.q, t - start)):
                    pred += self._ma_params[j] * residuals[t - start - j - 1]

            fitted[t - start] = pred
            residuals[t - start] = data[t] - pred

        return fitted

    def _integrate_fitted(
        self,
        fitted_diff: np.ndarray,
        original_data: np.ndarray,
        d: int
    ) -> np.ndarray:
        """Integrate fitted values back to original scale."""
        start = max(self.order.p, self.order.q)

        fitted = np.zeros(len(original_data))
        fitted[:d + start] = original_data[:d + start]

        for t in range(d + start, len(original_data)):
            fitted[t] = fitted_diff[t - d - start]

            for i in range(d):
                if t - i - 1 >= 0:
                    fitted[t] += original_data[t - i - 1]

        return fitted

    def _forecast_differenced(self, horizon: int) -> np.ndarray:
        """Forecast differenced series."""
        forecasts = np.zeros(horizon)

        if self._differenced_data is None:
            return forecasts

        history = list(self._differenced_data[-max(self.order.p, self.order.q):])
        residual_history = list(self._fitted.residuals[-self.order.q:] if self._fitted else [])

        for h in range(horizon):
            pred = self._constant

            if self.order.p > 0 and self._ar_params is not None:
                for i in range(self.order.p):
                    if len(history) > i:
                        pred += self._ar_params[i] * (history[-(i + 1)] - self._constant)

            forecasts[h] = pred

            history.append(pred)
            if len(history) > max(self.order.p, self.order.q):
                history.pop(0)

        return forecasts

    def _integrate_forecasts(
        self,
        forecasts_diff: np.ndarray,
        original_data: np.ndarray,
        d: int
    ) -> np.ndarray:
        """Integrate forecasts to original scale."""
        horizon = len(forecasts_diff)
        forecasts = np.zeros(horizon)

        last_values = list(original_data[-d:]) if d > 0 else []

        for h in range(horizon):
            if d == 0:
                forecasts[h] = forecasts_diff[h]
            elif d == 1:
                if h == 0:
                    forecasts[h] = forecasts_diff[h] + original_data[-1]
                else:
                    forecasts[h] = forecasts_diff[h] + forecasts[h - 1]
            else:
                forecasts[h] = forecasts_diff[h]
                for i in range(d):
                    if h - i - 1 >= 0:
                        forecasts[h] += forecasts[h - i - 1]
                    elif len(last_values) > 0:
                        forecasts[h] += last_values[-(i - h)]

        return forecasts

    def _compute_forecast_variance(self, horizon: int) -> np.ndarray:
        """Compute forecast error variance."""
        variance = np.zeros(horizon)

        psi = self._compute_psi_weights(horizon)

        for h in range(horizon):
            variance[h] = self._sigma2 * np.sum(psi[:h + 1] ** 2)

        return variance

    def _compute_psi_weights(self, n: int) -> np.ndarray:
        """Compute MA(infinity) weights."""
        psi = np.zeros(n)
        psi[0] = 1.0

        for j in range(1, n):
            if self._ar_params is not None:
                for i in range(min(j, len(self._ar_params))):
                    psi[j] += self._ar_params[i] * psi[j - i - 1]

            if self._ma_params is not None and j <= len(self._ma_params):
                psi[j] += self._ma_params[j - 1]

        return psi

    def _ppf_normal(self, p: float) -> float:
        """Percent point function for standard normal."""
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0

        a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00
        ]
        b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01
        ]
        c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00
        ]
        d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00
        ]

        p_low = 0.02425
        p_high = 1 - p_low

        if p < p_low:
            q = np.sqrt(-2 * np.log(p))
            return float(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )
        elif p <= p_high:
            q = p - 0.5
            r = q * q
            return float(
                (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
            )
        else:
            q = np.sqrt(-2 * np.log(1 - p))
            return float(
                -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
            )


class GARCHModel(BaseTimeSeriesModel):
    """GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model."""

    def __init__(
        self,
        order: GARCHOrder,
        name: str = "GARCH"
    ):
        """
        Initialize GARCH model.

        Args:
            order: GARCH order (p, q)
            name: Model name
        """
        super().__init__(name)
        self.order = order
        self._omega: float = 0.0
        self._alpha: Optional[np.ndarray] = None
        self._beta: Optional[np.ndarray] = None
        self._unconditional_var: float = 1.0
        self._returns: Optional[np.ndarray] = None

    async def fit(
        self,
        data: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6,
        **kwargs: Any
    ) -> FittedModel:
        """
        Fit GARCH model.

        Args:
            data: Returns data
            max_iter: Maximum iterations
            tol: Convergence tolerance
            **kwargs: Additional arguments

        Returns:
            FittedModel object
        """
        try:
            logger.info(f"Fitting GARCH{self.order.to_tuple()} model")

            self._returns = data.copy()

            self._omega, self._alpha, self._beta = self._estimate_parameters(
                data, max_iter, tol
            )

            conditional_var = self._compute_conditional_variance(data)

            standardized_residuals = data / np.sqrt(conditional_var)

            self._unconditional_var = self._omega / (
                1 - np.sum(self._alpha) - np.sum(self._beta)
            )

            n_params = 1 + self.order.p + self.order.q
            diagnostics = self._calculate_diagnostics(standardized_residuals, n_params)

            self._fitted = FittedModel(
                model_type=TimeSeriesModelType.GARCH,
                parameters={
                    "order": self.order.to_tuple(),
                    "omega": self._omega,
                    "alpha": self._alpha.tolist() if self._alpha is not None else [],
                    "beta": self._beta.tolist() if self._beta is not None else [],
                    "unconditional_var": self._unconditional_var
                },
                residuals=standardized_residuals,
                fitted_values=np.sqrt(conditional_var),
                diagnostics=diagnostics,
                training_data=data
            )

            self._is_fitted = True

            logger.info(
                f"GARCH fit complete: omega={self._omega:.6f}, "
                f"alpha={self._alpha}, beta={self._beta}"
            )

            return self._fitted

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {e}")
            raise

    async def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """
        Generate volatility forecasts.

        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level for intervals

        Returns:
            TimeSeriesForecast object
        """
        if not self._is_fitted or self._fitted is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            logger.info(f"Generating {horizon}-step GARCH volatility forecast")

            var_forecasts = self._forecast_variance(horizon)
            vol_forecasts = np.sqrt(var_forecasts)

            z_alpha = 1.96

            lower = vol_forecasts * (1 - z_alpha * 0.1)
            upper = vol_forecasts * (1 + z_alpha * 0.1)

            result = TimeSeriesForecast(
                forecast=vol_forecasts,
                confidence_lower=lower,
                confidence_upper=upper,
                confidence_level=confidence_level,
                model_name=self.name,
                forecast_horizon=horizon
            )

            logger.info(f"Volatility forecast generated: {vol_forecasts[:min(3, horizon)]}...")

            return result

        except Exception as e:
            logger.error(f"Error generating GARCH forecast: {e}")
            raise

    def _estimate_parameters(
        self,
        returns: np.ndarray,
        max_iter: int,
        tol: float
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Estimate GARCH parameters using variance targeting."""
        p, q = self.order.p, self.order.q

        sample_var = np.var(returns)

        alpha = np.ones(q) * 0.1 / q
        beta = np.ones(p) * 0.8 / p

        persistence = np.sum(alpha) + np.sum(beta)
        omega = sample_var * (1 - persistence)

        prev_params = np.concatenate([[omega], alpha, beta])

        for iteration in range(max_iter):
            sigma2 = self._compute_variance_series(returns, omega, alpha, beta)

            new_omega, new_alpha, new_beta = self._update_parameters(
                returns, sigma2, omega, alpha, beta
            )

            current_params = np.concatenate([[new_omega], new_alpha, new_beta])

            if np.max(np.abs(current_params - prev_params)) < tol:
                logger.debug(f"GARCH converged at iteration {iteration}")
                break

            omega, alpha, beta = new_omega, new_alpha, new_beta
            prev_params = current_params

        return omega, alpha, beta

    def _compute_variance_series(
        self,
        returns: np.ndarray,
        omega: float,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """Compute conditional variance series."""
        n = len(returns)
        p, q = len(beta), len(alpha)

        sigma2 = np.zeros(n)

        unconditional = omega / max(1 - np.sum(alpha) - np.sum(beta), 0.01)
        sigma2[:max(p, q)] = unconditional

        for t in range(max(p, q), n):
            sigma2[t] = omega

            for i in range(q):
                if t - i - 1 >= 0:
                    sigma2[t] += alpha[i] * returns[t - i - 1] ** 2

            for j in range(p):
                if t - j - 1 >= 0:
                    sigma2[t] += beta[j] * sigma2[t - j - 1]

            sigma2[t] = max(sigma2[t], 1e-10)

        return sigma2

    def _update_parameters(
        self,
        returns: np.ndarray,
        sigma2: np.ndarray,
        omega: float,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Update parameters using quasi-maximum likelihood."""
        n = len(returns)
        p, q = len(beta), len(alpha)
        start = max(p, q)

        epsilon2 = returns ** 2

        new_alpha = np.zeros(q)
        for i in range(q):
            numerator = np.sum(
                (epsilon2[start:] - sigma2[start:]) * epsilon2[start - i - 1:n - i - 1] /
                (sigma2[start:] ** 2)
            )
            denominator = np.sum(
                epsilon2[start - i - 1:n - i - 1] ** 2 / (sigma2[start:] ** 2)
            )
            if denominator > 0:
                new_alpha[i] = alpha[i] + 0.1 * numerator / denominator

        new_beta = np.zeros(p)
        for j in range(p):
            numerator = np.sum(
                (epsilon2[start:] - sigma2[start:]) * sigma2[start - j - 1:n - j - 1] /
                (sigma2[start:] ** 2)
            )
            denominator = np.sum(
                sigma2[start - j - 1:n - j - 1] ** 2 / (sigma2[start:] ** 2)
            )
            if denominator > 0:
                new_beta[j] = beta[j] + 0.1 * numerator / denominator

        new_alpha = np.clip(new_alpha, 0.001, 0.5)
        new_beta = np.clip(new_beta, 0.001, 0.99)

        persistence = np.sum(new_alpha) + np.sum(new_beta)
        if persistence >= 0.999:
            scale = 0.998 / persistence
            new_alpha *= scale
            new_beta *= scale

        sample_var = np.var(returns)
        new_omega = sample_var * (1 - np.sum(new_alpha) - np.sum(new_beta))
        new_omega = max(new_omega, 1e-10)

        return new_omega, new_alpha, new_beta

    def _compute_conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        """Compute conditional variance with fitted parameters."""
        if self._alpha is None or self._beta is None:
            return np.ones(len(returns)) * np.var(returns)

        return self._compute_variance_series(
            returns, self._omega, self._alpha, self._beta
        )

    def _forecast_variance(self, horizon: int) -> np.ndarray:
        """Forecast conditional variance."""
        if self._returns is None or self._alpha is None or self._beta is None:
            return np.ones(horizon) * self._unconditional_var

        forecasts = np.zeros(horizon)

        sigma2_history = list(self._compute_conditional_variance(self._returns)[-self.order.p:])
        epsilon2_history = list(self._returns[-self.order.q:] ** 2)

        for h in range(horizon):
            forecast = self._omega

            for i in range(self.order.q):
                if h == 0:
                    forecast += self._alpha[i] * epsilon2_history[-(i + 1)]
                else:
                    forecast += self._alpha[i] * forecasts[max(0, h - i - 1)]

            for j in range(self.order.p):
                if h - j - 1 < 0:
                    forecast += self._beta[j] * sigma2_history[-(j + 1 - h)]
                else:
                    forecast += self._beta[j] * forecasts[h - j - 1]

            forecasts[h] = forecast

        return forecasts


class ExponentialSmoothingModel(BaseTimeSeriesModel):
    """Exponential Smoothing (Holt-Winters) model."""

    def __init__(
        self,
        trend: TrendType = TrendType.ADDITIVE,
        seasonal: SeasonalityType = SeasonalityType.NONE,
        seasonal_period: int = 12,
        damped: bool = False,
        name: str = "ExponentialSmoothing"
    ):
        """
        Initialize Exponential Smoothing model.

        Args:
            trend: Type of trend component
            seasonal: Type of seasonal component
            seasonal_period: Seasonal period
            damped: Whether to use damped trend
            name: Model name
        """
        super().__init__(name)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self.damped = damped

        self._alpha: float = 0.0
        self._beta: float = 0.0
        self._gamma: float = 0.0
        self._phi: float = 1.0

        self._level: Optional[np.ndarray] = None
        self._trend_component: Optional[np.ndarray] = None
        self._seasonal_component: Optional[np.ndarray] = None
        self._data: Optional[np.ndarray] = None

    async def fit(
        self,
        data: np.ndarray,
        optimize: bool = True,
        **kwargs: Any
    ) -> FittedModel:
        """
        Fit Exponential Smoothing model.

        Args:
            data: Time series data
            optimize: Whether to optimize parameters
            **kwargs: Additional arguments

        Returns:
            FittedModel object
        """
        try:
            logger.info(
                f"Fitting {self.name} model: trend={self.trend.value}, "
                f"seasonal={self.seasonal.value}"
            )

            self._data = data.copy()
            n = len(data)

            if optimize:
                self._alpha, self._beta, self._gamma, self._phi = \
                    self._optimize_parameters(data)
            else:
                self._alpha = kwargs.get("alpha", 0.3)
                self._beta = kwargs.get("beta", 0.1)
                self._gamma = kwargs.get("gamma", 0.1)
                self._phi = kwargs.get("phi", 0.98)

            self._level = np.zeros(n)
            self._trend_component = np.zeros(n)
            self._seasonal_component = np.zeros(n + self.seasonal_period)

            self._initialize_components(data)

            fitted_values = self._compute_fitted_values(data)

            residuals = data - fitted_values

            n_params = self._count_parameters()
            diagnostics = self._calculate_diagnostics(residuals, n_params)

            self._fitted = FittedModel(
                model_type=TimeSeriesModelType.EXPONENTIAL_SMOOTHING,
                parameters={
                    "trend": self.trend.value,
                    "seasonal": self.seasonal.value,
                    "seasonal_period": self.seasonal_period,
                    "damped": self.damped,
                    "alpha": self._alpha,
                    "beta": self._beta,
                    "gamma": self._gamma,
                    "phi": self._phi
                },
                residuals=residuals,
                fitted_values=fitted_values,
                diagnostics=diagnostics,
                training_data=data
            )

            self._is_fitted = True

            logger.info(
                f"Exponential Smoothing fit complete: "
                f"alpha={self._alpha:.4f}, beta={self._beta:.4f}"
            )

            return self._fitted

        except Exception as e:
            logger.error(f"Error fitting Exponential Smoothing model: {e}")
            raise

    async def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """
        Generate forecasts.

        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level for intervals

        Returns:
            TimeSeriesForecast object
        """
        if not self._is_fitted or self._fitted is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            logger.info(f"Generating {horizon}-step Exponential Smoothing forecast")

            forecasts = self._generate_forecasts(horizon)

            residual_std = np.std(self._fitted.residuals)
            z_alpha = 1.96

            forecast_se = np.zeros(horizon)
            for h in range(horizon):
                if self.trend == TrendType.NONE:
                    c = 1
                elif self.damped:
                    c = sum(self._phi ** i for i in range(h + 1))
                else:
                    c = h + 1

                forecast_se[h] = residual_std * np.sqrt(1 + self._alpha ** 2 * c)

            lower = forecasts - z_alpha * forecast_se
            upper = forecasts + z_alpha * forecast_se

            result = TimeSeriesForecast(
                forecast=forecasts,
                confidence_lower=lower,
                confidence_upper=upper,
                confidence_level=confidence_level,
                model_name=self.name,
                forecast_horizon=horizon
            )

            logger.info(f"Forecast generated: {forecasts[:min(3, horizon)]}...")

            return result

        except Exception as e:
            logger.error(f"Error generating Exponential Smoothing forecast: {e}")
            raise

    def _optimize_parameters(
        self,
        data: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Optimize smoothing parameters using grid search."""
        best_sse = float("inf")
        best_params = (0.3, 0.1, 0.1, 0.98)

        alpha_range = np.arange(0.1, 1.0, 0.1)
        beta_range = np.arange(0.01, 0.5, 0.1) if self.trend != TrendType.NONE else [0.0]
        gamma_range = np.arange(0.01, 0.5, 0.1) if self.seasonal != SeasonalityType.NONE else [0.0]
        phi_range = np.arange(0.8, 1.0, 0.05) if self.damped else [1.0]

        for alpha in alpha_range:
            for beta in beta_range:
                for gamma in gamma_range:
                    for phi in phi_range:
                        try:
                            sse = self._compute_sse(data, alpha, beta, gamma, phi)
                            if sse < best_sse:
                                best_sse = sse
                                best_params = (alpha, beta, gamma, phi)
                        except Exception:
                            continue

        return best_params

    def _compute_sse(
        self,
        data: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        phi: float
    ) -> float:
        """Compute sum of squared errors for given parameters."""
        n = len(data)

        level = np.zeros(n)
        trend = np.zeros(n)
        seasonal = np.zeros(n + self.seasonal_period)

        self._initialize_components_with_params(data, level, trend, seasonal)

        sse = 0.0

        for t in range(self.seasonal_period, n):
            if self.seasonal == SeasonalityType.ADDITIVE:
                y_hat = level[t - 1] + phi * trend[t - 1] + seasonal[t - self.seasonal_period]
            elif self.seasonal == SeasonalityType.MULTIPLICATIVE:
                y_hat = (level[t - 1] + phi * trend[t - 1]) * seasonal[t - self.seasonal_period]
            else:
                y_hat = level[t - 1] + phi * trend[t - 1]

            error = data[t] - y_hat
            sse += error ** 2

            if self.seasonal == SeasonalityType.ADDITIVE:
                level[t] = alpha * (data[t] - seasonal[t - self.seasonal_period]) + \
                          (1 - alpha) * (level[t - 1] + phi * trend[t - 1])
                seasonal[t] = gamma * (data[t] - level[t]) + (1 - gamma) * seasonal[t - self.seasonal_period]
            elif self.seasonal == SeasonalityType.MULTIPLICATIVE:
                level[t] = alpha * (data[t] / max(seasonal[t - self.seasonal_period], 0.001)) + \
                          (1 - alpha) * (level[t - 1] + phi * trend[t - 1])
                seasonal[t] = gamma * (data[t] / max(level[t], 0.001)) + \
                             (1 - gamma) * seasonal[t - self.seasonal_period]
            else:
                level[t] = alpha * data[t] + (1 - alpha) * (level[t - 1] + phi * trend[t - 1])

            if self.trend == TrendType.ADDITIVE:
                trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * phi * trend[t - 1]
            elif self.trend == TrendType.MULTIPLICATIVE:
                if level[t - 1] > 0:
                    trend[t] = beta * (level[t] / level[t - 1]) + (1 - beta) * phi * trend[t - 1]

        return sse

    def _initialize_components(self, data: np.ndarray) -> None:
        """Initialize level, trend, and seasonal components."""
        if self._level is None or self._trend_component is None or self._seasonal_component is None:
            return

        self._initialize_components_with_params(
            data,
            self._level,
            self._trend_component,
            self._seasonal_component
        )

    def _initialize_components_with_params(
        self,
        data: np.ndarray,
        level: np.ndarray,
        trend: np.ndarray,
        seasonal: np.ndarray
    ) -> None:
        """Initialize components with given arrays."""
        m = self.seasonal_period

        if self.seasonal != SeasonalityType.NONE and len(data) >= 2 * m:
            season_means = np.array([
                np.mean(data[i * m:(i + 1) * m])
                for i in range(min(2, len(data) // m))
            ])

            level[m - 1] = np.mean(data[:m])

            if len(season_means) > 1:
                trend[m - 1] = (season_means[1] - season_means[0]) / m
            else:
                trend[m - 1] = 0

            if self.seasonal == SeasonalityType.ADDITIVE:
                for i in range(m):
                    seasonal[i] = data[i] - level[m - 1]
            else:
                for i in range(m):
                    if level[m - 1] > 0:
                        seasonal[i] = data[i] / level[m - 1]
                    else:
                        seasonal[i] = 1.0
        else:
            level[0] = data[0]
            trend[0] = data[1] - data[0] if len(data) > 1 else 0

            for t in range(1, min(m, len(data))):
                level[t] = level[0]
                trend[t] = trend[0]

    def _compute_fitted_values(self, data: np.ndarray) -> np.ndarray:
        """Compute fitted values."""
        if self._level is None or self._trend_component is None or self._seasonal_component is None:
            return np.zeros(len(data))

        n = len(data)
        fitted = np.zeros(n)
        m = self.seasonal_period

        start = m if self.seasonal != SeasonalityType.NONE else 1
        fitted[:start] = data[:start]

        for t in range(start, n):
            if self.seasonal == SeasonalityType.ADDITIVE:
                fitted[t] = self._level[t - 1] + self._phi * self._trend_component[t - 1] + \
                           self._seasonal_component[t - m]
            elif self.seasonal == SeasonalityType.MULTIPLICATIVE:
                fitted[t] = (self._level[t - 1] + self._phi * self._trend_component[t - 1]) * \
                           self._seasonal_component[t - m]
            else:
                fitted[t] = self._level[t - 1] + self._phi * self._trend_component[t - 1]

            if self.seasonal == SeasonalityType.ADDITIVE:
                self._level[t] = self._alpha * (data[t] - self._seasonal_component[t - m]) + \
                                (1 - self._alpha) * (self._level[t - 1] + self._phi * self._trend_component[t - 1])
                self._seasonal_component[t] = self._gamma * (data[t] - self._level[t]) + \
                                             (1 - self._gamma) * self._seasonal_component[t - m]
            elif self.seasonal == SeasonalityType.MULTIPLICATIVE:
                self._level[t] = self._alpha * (data[t] / max(self._seasonal_component[t - m], 0.001)) + \
                                (1 - self._alpha) * (self._level[t - 1] + self._phi * self._trend_component[t - 1])
                self._seasonal_component[t] = self._gamma * (data[t] / max(self._level[t], 0.001)) + \
                                             (1 - self._gamma) * self._seasonal_component[t - m]
            else:
                self._level[t] = self._alpha * data[t] + \
                                (1 - self._alpha) * (self._level[t - 1] + self._phi * self._trend_component[t - 1])

            if self.trend == TrendType.ADDITIVE:
                self._trend_component[t] = self._beta * (self._level[t] - self._level[t - 1]) + \
                                          (1 - self._beta) * self._phi * self._trend_component[t - 1]
            elif self.trend == TrendType.MULTIPLICATIVE:
                if self._level[t - 1] > 0:
                    self._trend_component[t] = self._beta * (self._level[t] / self._level[t - 1]) + \
                                              (1 - self._beta) * self._phi * self._trend_component[t - 1]

        return fitted

    def _generate_forecasts(self, horizon: int) -> np.ndarray:
        """Generate h-step ahead forecasts."""
        if self._level is None or self._trend_component is None or self._seasonal_component is None:
            return np.zeros(horizon)

        forecasts = np.zeros(horizon)
        n = len(self._level)
        m = self.seasonal_period

        last_level = self._level[n - 1]
        last_trend = self._trend_component[n - 1]

        for h in range(horizon):
            if self.damped:
                phi_sum = sum(self._phi ** i for i in range(1, h + 2))
            else:
                phi_sum = h + 1

            if self.seasonal == SeasonalityType.ADDITIVE:
                seasonal_idx = n - m + (h % m)
                forecasts[h] = last_level + phi_sum * last_trend + self._seasonal_component[seasonal_idx]
            elif self.seasonal == SeasonalityType.MULTIPLICATIVE:
                seasonal_idx = n - m + (h % m)
                forecasts[h] = (last_level + phi_sum * last_trend) * self._seasonal_component[seasonal_idx]
            else:
                forecasts[h] = last_level + phi_sum * last_trend

        return forecasts

    def _count_parameters(self) -> int:
        """Count number of model parameters."""
        n_params = 1

        if self.trend != TrendType.NONE:
            n_params += 1
            if self.damped:
                n_params += 1

        if self.seasonal != SeasonalityType.NONE:
            n_params += 1

        n_params += self.seasonal_period if self.seasonal != SeasonalityType.NONE else 0

        return n_params


class VARModel(BaseTimeSeriesModel):
    """Vector AutoRegression model for multivariate time series."""

    def __init__(
        self,
        order: int = 1,
        name: str = "VAR"
    ):
        """
        Initialize VAR model.

        Args:
            order: Model order (lag)
            name: Model name
        """
        super().__init__(name)
        self.order = order
        self._coefficients: Optional[np.ndarray] = None
        self._intercept: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._n_variables: int = 0
        self._data: Optional[np.ndarray] = None

    async def fit(
        self,
        data: np.ndarray,
        **kwargs: Any
    ) -> FittedModel:
        """
        Fit VAR model.

        Args:
            data: Multivariate time series (n_samples x n_variables)
            **kwargs: Additional arguments

        Returns:
            FittedModel object
        """
        try:
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self._n_variables = data.shape[1]
            self._data = data.copy()

            logger.info(f"Fitting VAR({self.order}) model with {self._n_variables} variables")

            self._coefficients, self._intercept = self._estimate_coefficients(data)

            fitted_values = self._compute_fitted_values(data)
            residuals = data[self.order:] - fitted_values

            self._sigma = np.cov(residuals.T)
            if self._sigma.ndim == 0:
                self._sigma = np.array([[self._sigma]])

            n_params = self._n_variables * (1 + self._n_variables * self.order)
            flat_residuals = residuals.flatten()
            diagnostics = self._calculate_diagnostics(flat_residuals, n_params)

            self._fitted = FittedModel(
                model_type=TimeSeriesModelType.VAR,
                parameters={
                    "order": self.order,
                    "n_variables": self._n_variables,
                    "coefficients": self._coefficients.tolist() if self._coefficients is not None else [],
                    "intercept": self._intercept.tolist() if self._intercept is not None else []
                },
                residuals=flat_residuals,
                fitted_values=fitted_values.flatten(),
                diagnostics=diagnostics,
                training_data=data
            )

            self._is_fitted = True

            logger.info(f"VAR fit complete: AIC={diagnostics.aic:.2f}")

            return self._fitted

        except Exception as e:
            logger.error(f"Error fitting VAR model: {e}")
            raise

    async def forecast(
        self,
        horizon: int,
        confidence_level: float = 0.95
    ) -> TimeSeriesForecast:
        """
        Generate VAR forecasts.

        Args:
            horizon: Forecast horizon
            confidence_level: Confidence level for intervals

        Returns:
            TimeSeriesForecast object
        """
        if not self._is_fitted or self._data is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            logger.info(f"Generating {horizon}-step VAR forecast")

            forecasts = self._generate_forecasts(horizon)

            z_alpha = 1.96

            if self._sigma is not None:
                se = np.sqrt(np.diag(self._sigma))
            else:
                se = np.ones(self._n_variables)

            forecast_se = np.outer(np.sqrt(np.arange(1, horizon + 1)), se)

            lower = forecasts - z_alpha * forecast_se
            upper = forecasts + z_alpha * forecast_se

            result = TimeSeriesForecast(
                forecast=forecasts.flatten(),
                confidence_lower=lower.flatten(),
                confidence_upper=upper.flatten(),
                confidence_level=confidence_level,
                model_name=self.name,
                forecast_horizon=horizon
            )

            logger.info(f"VAR forecast generated")

            return result

        except Exception as e:
            logger.error(f"Error generating VAR forecast: {e}")
            raise

    def _estimate_coefficients(
        self,
        data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate VAR coefficients using OLS."""
        n, k = data.shape
        p = self.order

        Y = data[p:]

        X = np.zeros((n - p, k * p + 1))
        X[:, 0] = 1

        for lag in range(1, p + 1):
            X[:, 1 + (lag - 1) * k:1 + lag * k] = data[p - lag:n - lag]

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros((k * p + 1, k))

        intercept = beta[0]
        coefficients = beta[1:].reshape(p, k, k)

        return coefficients, intercept

    def _compute_fitted_values(self, data: np.ndarray) -> np.ndarray:
        """Compute fitted values."""
        if self._coefficients is None or self._intercept is None:
            return np.zeros((len(data) - self.order, self._n_variables))

        n = len(data)
        fitted = np.zeros((n - self.order, self._n_variables))

        for t in range(self.order, n):
            fitted[t - self.order] = self._intercept.copy()

            for lag in range(self.order):
                fitted[t - self.order] += self._coefficients[lag] @ data[t - lag - 1]

        return fitted

    def _generate_forecasts(self, horizon: int) -> np.ndarray:
        """Generate multi-step forecasts."""
        if self._data is None or self._coefficients is None or self._intercept is None:
            return np.zeros((horizon, self._n_variables))

        forecasts = np.zeros((horizon, self._n_variables))

        history = list(self._data[-self.order:])

        for h in range(horizon):
            forecast = self._intercept.copy()

            for lag in range(self.order):
                if h - lag - 1 >= 0:
                    forecast += self._coefficients[lag] @ forecasts[h - lag - 1]
                else:
                    forecast += self._coefficients[lag] @ history[-(lag + 1 - h)]

            forecasts[h] = forecast

        return forecasts


def create_arima_model(
    p: int = 1,
    d: int = 1,
    q: int = 1,
    name: str = "ARIMA"
) -> ARIMAModel:
    """
    Factory function to create ARIMA model.

    Args:
        p: AR order
        d: Differencing order
        q: MA order
        name: Model name

    Returns:
        ARIMAModel instance
    """
    order = ARIMAOrder(p=p, d=d, q=q)
    return ARIMAModel(order=order, name=name)


def create_garch_model(
    p: int = 1,
    q: int = 1,
    name: str = "GARCH"
) -> GARCHModel:
    """
    Factory function to create GARCH model.

    Args:
        p: GARCH order
        q: ARCH order
        name: Model name

    Returns:
        GARCHModel instance
    """
    order = GARCHOrder(p=p, q=q)
    return GARCHModel(order=order, name=name)


def create_exponential_smoothing(
    trend: str = "additive",
    seasonal: str = "none",
    seasonal_period: int = 12,
    damped: bool = False,
    name: str = "ExponentialSmoothing"
) -> ExponentialSmoothingModel:
    """
    Factory function to create Exponential Smoothing model.

    Args:
        trend: Trend type
        seasonal: Seasonal type
        seasonal_period: Seasonal period
        damped: Whether to use damped trend
        name: Model name

    Returns:
        ExponentialSmoothingModel instance
    """
    trend_type = TrendType(trend)
    seasonal_type = SeasonalityType(seasonal)

    return ExponentialSmoothingModel(
        trend=trend_type,
        seasonal=seasonal_type,
        seasonal_period=seasonal_period,
        damped=damped,
        name=name
    )


def create_var_model(
    order: int = 1,
    name: str = "VAR"
) -> VARModel:
    """
    Factory function to create VAR model.

    Args:
        order: Model order
        name: Model name

    Returns:
        VARModel instance
    """
    return VARModel(order=order, name=name)
