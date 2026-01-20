"""
Cost Tracker for API cost monitoring with per-run and daily budget tracking.
~200 lines
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """Record of API cost."""
    timestamp: datetime
    run_id: str
    cost: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    operation: str = ""


class CostTracker:
    """
    Track API costs with per-run and daily budgets.

    Features:
    - Per-run budget tracking
    - Daily budget tracking
    - Degrade mode when budget exceeded
    - Cost history
    - Cost reporting
    """

    def __init__(
        self,
        daily_cap: float = 5.0,
        per_run_cap: float = 1.0
    ):
        """
        Initialize cost tracker.

        Args:
            daily_cap: Maximum daily spend in USD
            per_run_cap: Maximum spend per run_id in USD
        """
        self.daily_cap = daily_cap
        self.per_run_cap = per_run_cap

        # Current spending
        self.daily_spent = 0.0
        self.run_spent: Dict[str, float] = {}

        # Cost history
        self.cost_records: list[CostRecord] = []

        # Current date tracking
        self.current_date = date.today()

        # Degrade mode flag
        self.degrade_mode = False
        self.degrade_reason = ""

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"CostTracker initialized: daily_cap=${daily_cap}, per_run_cap=${per_run_cap}")

    def add_cost(
        self,
        cost: float,
        run_id: str,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        operation: str = ""
    ) -> bool:
        """
        Add cost and check against budgets.

        Args:
            cost: Cost in USD
            run_id: Run identifier
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            operation: Operation name

        Returns:
            True if cost added successfully (within budget)
        """
        with self._lock:
            # Check if new day
            self._check_daily_reset()

            # Check budgets before adding
            if self.daily_spent + cost > self.daily_cap:
                self.degrade_mode = True
                self.degrade_reason = f"Daily cap (${self.daily_cap}) exceeded"
                logger.warning(f"Daily budget exceeded: ${self.daily_spent:.4f} + ${cost:.4f} > ${self.daily_cap}")
                return False

            current_run_spent = self.run_spent.get(run_id, 0.0)
            if current_run_spent + cost > self.per_run_cap:
                self.degrade_mode = True
                self.degrade_reason = f"Per-run cap (${self.per_run_cap}) exceeded for run_id {run_id}"
                logger.warning(f"Per-run budget exceeded for {run_id}: ${current_run_spent:.4f} + ${cost:.4f} > ${self.per_run_cap}")
                return False

            # Add cost
            self.daily_spent += cost
            self.run_spent[run_id] = current_run_spent + cost

            # Record
            record = CostRecord(
                timestamp=datetime.now(),
                run_id=run_id,
                cost=cost,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                operation=operation
            )
            self.cost_records.append(record)

            logger.debug(f"Cost added: ${cost:.4f} for {run_id} ({operation})")
            return True

    def add(self, cost: float, run_id: str = "default") -> bool:
        """
        Simplified add method for backward compatibility.

        Args:
            cost: Cost in USD
            run_id: Run identifier

        Returns:
            True if cost added successfully
        """
        return self.add_cost(cost, run_id)

    def can_afford(self, cost: float, run_id: str) -> bool:
        """
        Check if a cost can be afforded without adding it.

        Args:
            cost: Proposed cost
            run_id: Run identifier

        Returns:
            True if within budget
        """
        with self._lock:
            self._check_daily_reset()

            if self.daily_spent + cost > self.daily_cap:
                return False

            current_run_spent = self.run_spent.get(run_id, 0.0)
            if current_run_spent + cost > self.per_run_cap:
                return False

            return True

    def is_degrade_mode(self) -> bool:
        """Check if in degrade mode."""
        return self.degrade_mode

    def get_degrade_reason(self) -> str:
        """Get reason for degrade mode."""
        return self.degrade_reason

    def get_daily_spent(self) -> float:
        """Get total daily spend."""
        with self._lock:
            self._check_daily_reset()
            return self.daily_spent

    def get_run_spent(self, run_id: str) -> float:
        """Get spend for specific run_id."""
        return self.run_spent.get(run_id, 0.0)

    def get_daily_remaining(self) -> float:
        """Get remaining daily budget."""
        with self._lock:
            self._check_daily_reset()
            return max(0.0, self.daily_cap - self.daily_spent)

    def get_run_remaining(self, run_id: str) -> float:
        """Get remaining budget for run_id."""
        current_spent = self.run_spent.get(run_id, 0.0)
        return max(0.0, self.per_run_cap - current_spent)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cost statistics."""
        with self._lock:
            self._check_daily_reset()

            return {
                "daily_cap": self.daily_cap,
                "per_run_cap": self.per_run_cap,
                "daily_spent": round(self.daily_spent, 4),
                "daily_remaining": round(self.get_daily_remaining(), 4),
                "total_runs": len(self.run_spent),
                "degrade_mode": self.degrade_mode,
                "degrade_reason": self.degrade_reason,
                "total_records": len(self.cost_records),
                "current_date": self.current_date.isoformat()
            }

    def get_run_stats(self, run_id: str) -> Dict[str, Any]:
        """Get statistics for specific run_id."""
        spent = self.get_run_spent(run_id)
        remaining = self.get_run_remaining(run_id)

        # Filter records for this run
        run_records = [r for r in self.cost_records if r.run_id == run_id]

        return {
            "run_id": run_id,
            "spent": round(spent, 4),
            "remaining": round(remaining, 4),
            "cap": self.per_run_cap,
            "num_operations": len(run_records),
            "within_budget": spent <= self.per_run_cap
        }

    def get_cost_history(
        self,
        run_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> list[Dict[str, Any]]:
        """
        Get cost history.

        Args:
            run_id: Filter by run_id
            limit: Limit number of records

        Returns:
            List of cost records
        """
        records = self.cost_records

        if run_id:
            records = [r for r in records if r.run_id == run_id]

        if limit:
            records = records[-limit:]

        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "run_id": r.run_id,
                "cost": round(r.cost, 6),
                "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "operation": r.operation
            }
            for r in records
        ]

    def reset_degrade_mode(self):
        """Manually reset degrade mode (use with caution)."""
        with self._lock:
            self.degrade_mode = False
            self.degrade_reason = ""
            logger.info("Degrade mode manually reset")

    def _check_daily_reset(self):
        """Check if we've entered a new day and reset daily counters."""
        today = date.today()

        if today > self.current_date:
            logger.info(f"New day detected, resetting daily counters. Previous spend: ${self.daily_spent:.4f}")
            self.current_date = today
            self.daily_spent = 0.0
            self.run_spent.clear()
            self.degrade_mode = False
            self.degrade_reason = ""

    def clear_history(self):
        """Clear cost history (keeps current counters)."""
        with self._lock:
            self.cost_records.clear()
            logger.info("Cost history cleared")
