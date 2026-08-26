"""
MissionMind safety package — deterministic hard-constraint enforcement.

This package contains no AI/LLM code.  It is a pure-Python rule engine
that enforces hard constraints on mission plans before they are executed.

Exports
-------
SafetyValidator   The validator class.  Call ``validate(plan, rover_state)``.
ValidationResult  Dataclass returned by ``validate()``.
"""

from missionmind.safety.validator import SafetyValidator, ValidationResult

__all__ = ["SafetyValidator", "ValidationResult"]
