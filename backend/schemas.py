"""Pydantic response schemas of the GridSentinel API.

Every payload separates four kinds of information, never mixed:
  ``ground_truth``  - what the simulator actually did (labels, true values)
  ``ai``            - Random Forest predictions
  ``physics``       - state-estimation / residual verification
  ``mitigation``    - operator-supervised recommendations
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class StreamStatus(BaseModel):
    state: str = Field(description="running | stale | stopped | completed | failed | interrupted | no_data")
    live: bool
    message: str
    run_id: Optional[str] = None
    mode: Optional[str] = None
    last_step: Optional[int] = None
    seconds_since_last_measurement: Optional[float] = None
    stale_after_s: Optional[float] = None
    interval_s: Optional[float] = None


class Health(BaseModel):
    api: str
    database: str
    database_path: str
    models_loaded: bool
    model_error: Optional[str] = None
    model_info: Optional[dict[str, Any]] = None
    stream: StreamStatus
    monitor: dict[str, Any]
    server_time: float


class BusState(BaseModel):
    bus: int
    type: str
    is_generator: bool
    received: dict[str, Optional[float]]
    ground_truth: dict[str, Any]
    ai: Optional[dict[str, Any]] = None
    voltage_limits: list[float]
    voltage_violation: bool


class LineState(BaseModel):
    line: int
    from_bus: int
    to_bus: int
    rating_mva: Optional[float]
    received: dict[str, Optional[float]]
    ground_truth: dict[str, Optional[float]]
    overloaded: Optional[bool]


class ErrorBody(BaseModel):
    error: str
    detail: str
    hint: Optional[str] = None
