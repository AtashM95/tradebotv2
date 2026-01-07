"""
State Machine Module for Ultimate Trading Bot v2.2.

This module provides a generic finite state machine implementation
for managing system and component states with transitions, guards,
and actions.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

from pydantic import BaseModel, Field

from src.utils.exceptions import ValidationError
from src.utils.helpers import generate_uuid
from src.utils.date_utils import now_utc


logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=Enum)


class TransitionResult(BaseModel):
    """Result of a state transition attempt."""

    success: bool = Field(default=False)
    from_state: str
    to_state: str
    timestamp: datetime = Field(default_factory=now_utc)
    message: str = Field(default="")
    data: dict = Field(default_factory=dict)


class StateTransition(BaseModel):
    """Definition of a state transition."""

    transition_id: str = Field(default_factory=generate_uuid)
    name: str
    from_state: str
    to_state: str
    description: str = Field(default="")

    auto_trigger: bool = Field(default=False)
    requires_confirmation: bool = Field(default=False)

    created_at: datetime = Field(default_factory=now_utc)


class StateHistory(BaseModel):
    """History of state transitions."""

    entries: list[TransitionResult] = Field(default_factory=list)
    max_entries: int = Field(default=1000)

    def add(self, result: TransitionResult) -> None:
        """Add a transition result to history."""
        self.entries.append(result)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def get_recent(self, count: int = 10) -> list[TransitionResult]:
        """Get recent transition results."""
        return self.entries[-count:]

    def get_transitions_from(self, state: str) -> list[TransitionResult]:
        """Get all transitions from a specific state."""
        return [e for e in self.entries if e.from_state == state]

    def get_transitions_to(self, state: str) -> list[TransitionResult]:
        """Get all transitions to a specific state."""
        return [e for e in self.entries if e.to_state == state]

    def clear(self) -> None:
        """Clear history."""
        self.entries.clear()


class StateMachine(Generic[StateT]):
    """
    Generic finite state machine implementation.

    This class provides:
    - State management with type safety
    - Transition definitions with guards
    - Entry/exit actions
    - Transition history
    - Event-driven state changes
    """

    def __init__(
        self,
        name: str,
        initial_state: StateT,
        state_enum: type[StateT],
    ) -> None:
        """
        Initialize StateMachine.

        Args:
            name: State machine name
            initial_state: Initial state
            state_enum: Enum class for states
        """
        self._name = name
        self._state_enum = state_enum
        self._current_state = initial_state
        self._previous_state: Optional[StateT] = None

        self._transitions: dict[tuple[str, str], StateTransition] = {}

        self._guards: dict[tuple[str, str], list[Callable[..., bool]]] = {}

        self._entry_actions: dict[str, list[Callable[..., Any]]] = {}
        self._exit_actions: dict[str, list[Callable[..., Any]]] = {}
        self._transition_actions: dict[tuple[str, str], list[Callable[..., Any]]] = {}

        self._state_callbacks: dict[str, list[Callable[[StateT], None]]] = {}
        self._any_transition_callbacks: list[Callable[[StateT, StateT], None]] = []

        self._history = StateHistory()

        self._lock = asyncio.Lock()
        self._state_changed = asyncio.Event()

        logger.info(f"StateMachine '{name}' initialized with state: {initial_state}")

    @property
    def name(self) -> str:
        """Get state machine name."""
        return self._name

    @property
    def current_state(self) -> StateT:
        """Get current state."""
        return self._current_state

    @property
    def previous_state(self) -> Optional[StateT]:
        """Get previous state."""
        return self._previous_state

    @property
    def history(self) -> StateHistory:
        """Get transition history."""
        return self._history

    @property
    def state_changed_event(self) -> asyncio.Event:
        """Get state changed event."""
        return self._state_changed

    def define_transition(
        self,
        name: str,
        from_state: StateT,
        to_state: StateT,
        description: str = "",
        auto_trigger: bool = False,
        requires_confirmation: bool = False,
    ) -> StateTransition:
        """
        Define a valid state transition.

        Args:
            name: Transition name
            from_state: Source state
            to_state: Target state
            description: Transition description
            auto_trigger: Whether transition can be auto-triggered
            requires_confirmation: Whether transition requires confirmation

        Returns:
            Created transition definition
        """
        key = (from_state.value, to_state.value)

        transition = StateTransition(
            name=name,
            from_state=from_state.value,
            to_state=to_state.value,
            description=description,
            auto_trigger=auto_trigger,
            requires_confirmation=requires_confirmation,
        )

        self._transitions[key] = transition

        logger.debug(
            f"Defined transition '{name}': {from_state.value} -> {to_state.value}"
        )

        return transition

    def add_guard(
        self,
        from_state: StateT,
        to_state: StateT,
        guard: Callable[..., bool]
    ) -> None:
        """
        Add a guard condition for a transition.

        Args:
            from_state: Source state
            to_state: Target state
            guard: Guard function returning bool
        """
        key = (from_state.value, to_state.value)

        if key not in self._guards:
            self._guards[key] = []

        self._guards[key].append(guard)
        logger.debug(f"Added guard for {from_state.value} -> {to_state.value}")

    def add_entry_action(
        self,
        state: StateT,
        action: Callable[..., Any]
    ) -> None:
        """
        Add an entry action for a state.

        Args:
            state: Target state
            action: Action to execute on entry
        """
        state_key = state.value

        if state_key not in self._entry_actions:
            self._entry_actions[state_key] = []

        self._entry_actions[state_key].append(action)
        logger.debug(f"Added entry action for state {state_key}")

    def add_exit_action(
        self,
        state: StateT,
        action: Callable[..., Any]
    ) -> None:
        """
        Add an exit action for a state.

        Args:
            state: Source state
            action: Action to execute on exit
        """
        state_key = state.value

        if state_key not in self._exit_actions:
            self._exit_actions[state_key] = []

        self._exit_actions[state_key].append(action)
        logger.debug(f"Added exit action for state {state_key}")

    def add_transition_action(
        self,
        from_state: StateT,
        to_state: StateT,
        action: Callable[..., Any]
    ) -> None:
        """
        Add an action for a specific transition.

        Args:
            from_state: Source state
            to_state: Target state
            action: Action to execute during transition
        """
        key = (from_state.value, to_state.value)

        if key not in self._transition_actions:
            self._transition_actions[key] = []

        self._transition_actions[key].append(action)
        logger.debug(f"Added transition action for {from_state.value} -> {to_state.value}")

    def on_state(
        self,
        state: StateT,
        callback: Callable[[StateT], None]
    ) -> None:
        """
        Register a callback for when a specific state is entered.

        Args:
            state: Target state
            callback: Callback function
        """
        state_key = state.value

        if state_key not in self._state_callbacks:
            self._state_callbacks[state_key] = []

        self._state_callbacks[state_key].append(callback)

    def on_any_transition(
        self,
        callback: Callable[[StateT, StateT], None]
    ) -> None:
        """
        Register a callback for any state transition.

        Args:
            callback: Callback function receiving (from_state, to_state)
        """
        self._any_transition_callbacks.append(callback)

    def can_transition(
        self,
        to_state: StateT,
        context: Optional[dict] = None
    ) -> tuple[bool, str]:
        """
        Check if a transition is possible.

        Args:
            to_state: Target state
            context: Optional context for guards

        Returns:
            Tuple of (can_transition, reason)
        """
        from_key = self._current_state.value
        to_key = to_state.value
        key = (from_key, to_key)

        if key not in self._transitions:
            return False, f"No transition defined from {from_key} to {to_key}"

        guards = self._guards.get(key, [])
        context = context or {}

        for guard in guards:
            try:
                if asyncio.iscoroutinefunction(guard):
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        return False, "Cannot check async guard synchronously"
                    result = loop.run_until_complete(guard(**context))
                else:
                    result = guard(**context)

                if not result:
                    return False, f"Guard condition failed: {guard.__name__}"
            except Exception as e:
                return False, f"Guard error: {e}"

        return True, "Transition allowed"

    async def transition_to(
        self,
        to_state: StateT,
        context: Optional[dict] = None,
        force: bool = False
    ) -> TransitionResult:
        """
        Attempt to transition to a new state.

        Args:
            to_state: Target state
            context: Optional context data
            force: Force transition without guards

        Returns:
            Transition result
        """
        async with self._lock:
            from_state = self._current_state
            from_key = from_state.value
            to_key = to_state.value
            key = (from_key, to_key)
            context = context or {}

            if from_state == to_state:
                return TransitionResult(
                    success=True,
                    from_state=from_key,
                    to_state=to_key,
                    message="Already in target state",
                )

            if not force:
                can_trans, reason = await self._check_guards_async(key, context)
                if not can_trans:
                    result = TransitionResult(
                        success=False,
                        from_state=from_key,
                        to_state=to_key,
                        message=reason,
                    )
                    self._history.add(result)
                    return result

            try:
                await self._execute_exit_actions(from_key, context)

                await self._execute_transition_actions(key, context)

                self._previous_state = from_state
                self._current_state = to_state

                await self._execute_entry_actions(to_key, context)

                await self._notify_state_change(from_state, to_state)

                self._state_changed.set()
                self._state_changed.clear()

                result = TransitionResult(
                    success=True,
                    from_state=from_key,
                    to_state=to_key,
                    message="Transition successful",
                    data=context,
                )

                self._history.add(result)

                logger.info(
                    f"StateMachine '{self._name}': {from_key} -> {to_key}"
                )

                return result

            except Exception as e:
                logger.error(f"Error during transition: {e}")

                result = TransitionResult(
                    success=False,
                    from_state=from_key,
                    to_state=to_key,
                    message=f"Transition error: {e}",
                )

                self._history.add(result)
                return result

    async def _check_guards_async(
        self,
        key: tuple[str, str],
        context: dict
    ) -> tuple[bool, str]:
        """Check all guards for a transition asynchronously."""
        if key not in self._transitions:
            return False, f"No transition defined: {key[0]} -> {key[1]}"

        guards = self._guards.get(key, [])

        for guard in guards:
            try:
                if asyncio.iscoroutinefunction(guard):
                    result = await guard(**context)
                else:
                    result = guard(**context)

                if not result:
                    return False, f"Guard failed: {guard.__name__}"
            except Exception as e:
                return False, f"Guard error: {e}"

        return True, "All guards passed"

    async def _execute_exit_actions(
        self,
        state_key: str,
        context: dict
    ) -> None:
        """Execute exit actions for a state."""
        actions = self._exit_actions.get(state_key, [])

        for action in actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(**context)
                else:
                    action(**context)
            except Exception as e:
                logger.error(f"Error in exit action: {e}")

    async def _execute_entry_actions(
        self,
        state_key: str,
        context: dict
    ) -> None:
        """Execute entry actions for a state."""
        actions = self._entry_actions.get(state_key, [])

        for action in actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(**context)
                else:
                    action(**context)
            except Exception as e:
                logger.error(f"Error in entry action: {e}")

    async def _execute_transition_actions(
        self,
        key: tuple[str, str],
        context: dict
    ) -> None:
        """Execute transition-specific actions."""
        actions = self._transition_actions.get(key, [])

        for action in actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(**context)
                else:
                    action(**context)
            except Exception as e:
                logger.error(f"Error in transition action: {e}")

    async def _notify_state_change(
        self,
        from_state: StateT,
        to_state: StateT
    ) -> None:
        """Notify callbacks of state change."""
        to_key = to_state.value

        for callback in self._state_callbacks.get(to_key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(to_state)
                else:
                    callback(to_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")

        for callback in self._any_transition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(from_state, to_state)
                else:
                    callback(from_state, to_state)
            except Exception as e:
                logger.error(f"Error in transition callback: {e}")

    def get_available_transitions(self) -> list[StateTransition]:
        """Get available transitions from current state."""
        from_key = self._current_state.value
        return [
            trans for (fk, _), trans in self._transitions.items()
            if fk == from_key
        ]

    def get_all_transitions(self) -> list[StateTransition]:
        """Get all defined transitions."""
        return list(self._transitions.values())

    def get_states(self) -> list[StateT]:
        """Get all possible states."""
        return list(self._state_enum)

    def is_in_state(self, state: StateT) -> bool:
        """Check if machine is in a specific state."""
        return self._current_state == state

    def is_in_any_state(self, states: list[StateT]) -> bool:
        """Check if machine is in any of the specified states."""
        return self._current_state in states

    async def wait_for_state(
        self,
        state: StateT,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Wait for the machine to enter a specific state.

        Args:
            state: Target state
            timeout: Maximum wait time in seconds

        Returns:
            True if state reached, False if timeout
        """
        if self._current_state == state:
            return True

        try:
            while self._current_state != state:
                await asyncio.wait_for(
                    self._state_changed.wait(),
                    timeout=timeout
                )

                if self._current_state == state:
                    return True

        except asyncio.TimeoutError:
            return False

        return True

    async def reset(self, initial_state: Optional[StateT] = None) -> None:
        """
        Reset the state machine.

        Args:
            initial_state: State to reset to
        """
        async with self._lock:
            if initial_state:
                self._current_state = initial_state
            self._previous_state = None
            self._history.clear()

            logger.info(
                f"StateMachine '{self._name}' reset to {self._current_state}"
            )

    def get_state_info(self) -> dict:
        """Get current state information."""
        return {
            "name": self._name,
            "current_state": self._current_state.value,
            "previous_state": self._previous_state.value if self._previous_state else None,
            "available_transitions": [
                {
                    "name": t.name,
                    "to_state": t.to_state,
                }
                for t in self.get_available_transitions()
            ],
            "transition_count": len(self._history.entries),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StateMachine(name='{self._name}', "
            f"state={self._current_state.value})"
        )


class CompositeStateMachine:
    """
    Manages multiple related state machines.

    This class provides:
    - Coordination between multiple state machines
    - Hierarchical state management
    - Cross-machine transitions
    """

    def __init__(self, name: str) -> None:
        """
        Initialize CompositeStateMachine.

        Args:
            name: Composite machine name
        """
        self._name = name
        self._machines: dict[str, StateMachine] = {}
        self._dependencies: dict[str, list[tuple[str, Any, Any]]] = {}

        logger.info(f"CompositeStateMachine '{name}' initialized")

    def add_machine(
        self,
        machine_id: str,
        machine: StateMachine
    ) -> None:
        """Add a state machine to the composite."""
        self._machines[machine_id] = machine
        logger.debug(f"Added machine '{machine_id}' to composite")

    def get_machine(self, machine_id: str) -> Optional[StateMachine]:
        """Get a state machine by ID."""
        return self._machines.get(machine_id)

    def add_dependency(
        self,
        machine_id: str,
        depends_on: str,
        required_state: Any,
        blocked_state: Any
    ) -> None:
        """
        Add a dependency between machines.

        Args:
            machine_id: Dependent machine
            depends_on: Machine to depend on
            required_state: Required state of depends_on
            blocked_state: State that blocks machine_id
        """
        if machine_id not in self._dependencies:
            self._dependencies[machine_id] = []

        self._dependencies[machine_id].append(
            (depends_on, required_state, blocked_state)
        )

    def check_dependencies(self, machine_id: str) -> tuple[bool, str]:
        """Check if a machine's dependencies are met."""
        deps = self._dependencies.get(machine_id, [])

        for dep_id, required_state, blocked_state in deps:
            dep_machine = self._machines.get(dep_id)
            if not dep_machine:
                continue

            if dep_machine.is_in_state(blocked_state):
                return False, f"Blocked by {dep_id} in state {blocked_state}"

            if not dep_machine.is_in_state(required_state):
                return False, f"Requires {dep_id} in state {required_state}"

        return True, "All dependencies met"

    def get_states(self) -> dict[str, str]:
        """Get current states of all machines."""
        return {
            machine_id: machine.current_state.value
            for machine_id, machine in self._machines.items()
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CompositeStateMachine(name='{self._name}', "
            f"machines={list(self._machines.keys())})"
        )
