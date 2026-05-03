"""
temporal_nudge.py — Adaptive nudging with Time-to-Fill (TTF) tracking.

Wraps the base NudgeEngine and adds temporal awareness:
  • Tracks how many poll cycles each signal has been active without
    a corresponding visual state change (i.e., the offer hasn't filled).
  • Escalates the nudge offset progressively when offers stagnate.
  • Implements a configurable escalation curve (linear, geometric, or
    capped-step) to find the current liquidity floor.
  • Provides de-escalation: once a fill is confirmed, the offset resets
    to baseline for that asset.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

from data_poller import AssetConfig, NudgeEngine, NudgeResult, VelocityAnalyzer

log = logging.getLogger(__name__)


class EscalationMode(Enum):
    LINEAR = auto()       # offset += step per stale cycle
    GEOMETRIC = auto()    # offset *= factor per stale cycle
    CAPPED_STEP = auto()  # offset += step, capped at max


@dataclass
class TTFConfig:
    """Configuration for Time-to-Fill adaptive nudging."""

    # After this many stale cycles, begin escalation
    stale_threshold_cycles: int = 4

    # Escalation parameters
    mode: EscalationMode = EscalationMode.CAPPED_STEP
    linear_step_pct: float = 0.15        # Added per stale cycle (LINEAR mode)
    geometric_factor: float = 1.12       # Multiplied per stale cycle (GEOMETRIC)
    capped_step_pct: float = 0.2         # Step size (CAPPED_STEP)
    max_offset_pct: float = 5.0          # Hard ceiling — never nudge beyond this

    # De-escalation
    reset_on_fill: bool = True           # Snap back to baseline on confirmed fill
    decay_rate_per_cycle: float = 0.3    # Gradual decay toward baseline after fill

    # Logging
    log_escalations: bool = True


@dataclass
class AssetTTFState:
    """Per-asset temporal tracking state."""
    current_signal: str = "HOLD"
    signal_start_cycle: int = 0
    stale_cycles: int = 0              # Cycles at same signal without fill
    current_offset_bonus_pct: float = 0.0
    total_escalations: int = 0
    last_fill_at: float = 0.0
    last_escalation_at: float = 0.0

    @property
    def is_stale(self) -> bool:
        return self.stale_cycles > 0


class TemporalNudgeEngine:
    """
    Wraps NudgeEngine with time-to-fill awareness.

    Usage:
        base_nudge = NudgeEngine(strategy_config)
        temporal = TemporalNudgeEngine(base_nudge, ttf_config)

        # Each poll cycle:
        result = temporal.evaluate(asset, analyzer, cycle_number)

        # When VisionEngine confirms a slot state changed (fill detected):
        temporal.report_fill(asset_id)
    """

    def __init__(
        self,
        base_engine: NudgeEngine,
        ttf_config: TTFConfig | None = None,
    ):
        self.base = base_engine
        self.cfg = ttf_config or TTFConfig()
        self._states: dict[int, AssetTTFState] = {}
        self._global_cycle: int = 0

    # ── State access ─────────────────────────────────────────────────────

    def _get_state(self, asset_id: int) -> AssetTTFState:
        if asset_id not in self._states:
            self._states[asset_id] = AssetTTFState()
        return self._states[asset_id]

    def get_ttf_status(self, asset_id: int) -> dict:
        s = self._get_state(asset_id)
        return {
            "signal": s.current_signal,
            "stale_cycles": s.stale_cycles,
            "offset_bonus_pct": round(s.current_offset_bonus_pct, 3),
            "total_escalations": s.total_escalations,
            "last_fill_ago_sec": round(time.time() - s.last_fill_at, 1) if s.last_fill_at else None,
        }

    # ── Escalation logic ────────────────────────────────────────────────

    def _compute_escalated_offset(self, state: AssetTTFState) -> float:
        """
        Compute the additional offset bonus based on how many stale
        cycles have elapsed beyond the threshold.
        """
        excess = max(0, state.stale_cycles - self.cfg.stale_threshold_cycles)
        if excess == 0:
            return 0.0

        mode = self.cfg.mode

        if mode == EscalationMode.LINEAR:
            bonus = excess * self.cfg.linear_step_pct

        elif mode == EscalationMode.GEOMETRIC:
            bonus = (self.cfg.geometric_factor ** excess - 1.0) * 100.0
            # Convert from multiplicative to additive percentage
            # e.g., 1.12^3 - 1 ≈ 0.405 → 0.405% additional offset
            bonus = (self.cfg.geometric_factor ** excess - 1.0) * self.base.nudge_pct * 100

        elif mode == EscalationMode.CAPPED_STEP:
            bonus = excess * self.cfg.capped_step_pct

        else:
            bonus = 0.0

        return min(bonus, self.cfg.max_offset_pct)

    def _apply_escalation(self, state: AssetTTFState) -> float:
        """Update the state's offset bonus and return the new value."""
        new_bonus = self._compute_escalated_offset(state)
        old_bonus = state.current_offset_bonus_pct

        if new_bonus > old_bonus:
            state.current_offset_bonus_pct = new_bonus
            state.total_escalations += 1
            state.last_escalation_at = time.time()
            if self.cfg.log_escalations:
                log.info(
                    "TTF escalation: asset=%d stale=%d bonus=%.2f%% → %.2f%%",
                    0, state.stale_cycles, old_bonus, new_bonus,
                )
        elif new_bonus < old_bonus:
            # Gradual decay (shouldn't normally happen during escalation)
            state.current_offset_bonus_pct = new_bonus

        return state.current_offset_bonus_pct

    # ── Core evaluation ──────────────────────────────────────────────────

    def evaluate(
        self,
        asset: AssetConfig,
        analyzer: VelocityAnalyzer,
        cycle: int | None = None,
    ) -> NudgeResult | None:
        """
        Evaluate an asset with temporal awareness.

        1. Run the base NudgeEngine to get the raw signal.
        2. Track signal continuity (same signal = stale).
        3. Escalate the offset if stale beyond threshold.
        4. Recompute buy/sell targets with the escalated offset.
        """
        if cycle is not None:
            self._global_cycle = cycle

        # Base evaluation
        result = self.base.evaluate(asset, analyzer)
        if result is None:
            return None

        state = self._get_state(asset.id)

        # ── Signal continuity tracking ──
        if result.signal != state.current_signal:
            # Signal changed — reset staleness
            state.current_signal = result.signal
            state.signal_start_cycle = self._global_cycle
            state.stale_cycles = 0
            # Don't reset offset immediately — let decay handle it
        else:
            # Same signal persists
            state.stale_cycles = self._global_cycle - state.signal_start_cycle

        # ── Only escalate actionable signals ──
        if result.signal not in ("BUY", "SELL"):
            # HOLD / SKIP — apply gradual decay toward baseline
            if state.current_offset_bonus_pct > 0:
                state.current_offset_bonus_pct = max(
                    0.0,
                    state.current_offset_bonus_pct - self.cfg.decay_rate_per_cycle,
                )
            return result

        # ── Escalation ──
        offset_bonus = self._apply_escalation(state)

        if offset_bonus <= 0:
            return result  # No modification needed

        # ── Recompute targets with escalated offset ──
        sma = result.moving_avg
        base_nudge_pct = self.base.nudge_pct + (asset.price_buffer_pct / 100.0)
        escalated_nudge_pct = base_nudge_pct + (offset_bonus / 100.0)

        if result.signal == "BUY":
            # More aggressive buying: increase the bid (reduce discount)
            # When stale, we're not getting filled — need to offer more
            new_buy = int(sma * (1.0 - max(0, base_nudge_pct - offset_bonus / 100.0)))
            new_sell = result.suggested_sell  # Keep sell target unchanged
        elif result.signal == "SELL":
            # More aggressive selling: decrease the ask
            new_sell = int(sma * (1.0 + max(0, base_nudge_pct - offset_bonus / 100.0)))
            new_buy = result.suggested_buy  # Keep buy target unchanged
        else:
            new_buy = result.suggested_buy
            new_sell = result.suggested_sell

        # Ensure sell still clears break-even
        be_floor = int(new_buy * (1.0 + self.base.tax_pct + self.base.margin_pct))
        if new_sell <= be_floor:
            new_sell = be_floor + 1

        return NudgeResult(
            item_id=result.item_id,
            item_name=result.item_name,
            moving_avg=result.moving_avg,
            velocity=result.velocity,
            liquidity_score=result.liquidity_score,
            suggested_buy=new_buy,
            suggested_sell=new_sell,
            break_even_floor=be_floor,
            spread_pct=result.spread_pct,
            signal=result.signal,
        )

    # ── Fill reporting ───────────────────────────────────────────────────

    def report_fill(self, asset_id: int):
        """
        Called when VisionEngine confirms a slot state change (offer filled).
        Resets the temporal state for this asset.
        """
        state = self._get_state(asset_id)
        state.last_fill_at = time.time()
        state.stale_cycles = 0
        state.signal_start_cycle = self._global_cycle

        if self.cfg.reset_on_fill:
            old_bonus = state.current_offset_bonus_pct
            state.current_offset_bonus_pct = 0.0
            if old_bonus > 0:
                log.info(
                    "Fill confirmed for asset %d. Offset bonus reset (was %.2f%%).",
                    asset_id, old_bonus,
                )
        else:
            # Gradual decay instead of hard reset
            state.current_offset_bonus_pct *= 0.5

    # ── Bulk operations ──────────────────────────────────────────────────

    def evaluate_all(
        self,
        assets: list[AssetConfig],
        analyzer: VelocityAnalyzer,
    ) -> list[NudgeResult]:
        """Convenience: evaluate all assets in one call, incrementing the cycle."""
        self._global_cycle += 1
        results = []
        for asset in assets:
            r = self.evaluate(asset, analyzer, self._global_cycle)
            if r:
                results.append(r)
        return results

    def status_summary(self) -> list[dict]:
        """Return TTF status for all tracked assets."""
        return [
            {"asset_id": aid, **self.get_ttf_status(aid)}
            for aid in sorted(self._states.keys())
        ]
