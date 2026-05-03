"""
CoordinateMapper — Adaptive absolute→relative coordinate translation with
state continuity and desync recovery.

Problem:
  VisionEngine detects targets in absolute screen coordinates.
  SerialController operates in relative HID deltas.
  Over time, accumulated rounding errors and missed HID reports cause
  the "Last Known Position" (LKP) to drift from the true cursor location.

Solution:
  • Track LKP after every issued movement.
  • Provide (dx, dy) vector computation from LKP → target.
  • Detect drift via a CV-based cursor locator (optional).
  • Execute a "re-zero" routine by driving to a known anchor point
    when drift exceeds a configurable threshold.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SyncState(Enum):
    SYNCED = auto()
    DRIFTED = auto()         # Drift detected but within recovery range
    DESYNCED = auto()        # LKP is unreliable — re-zero required
    REZEROING = auto()       # Currently executing a re-zero routine


@dataclass
class CursorState:
    x: int = 0
    y: int = 0
    confidence: float = 1.0  # 1.0 = just re-zeroed, decays with each move
    last_verified_at: float = 0.0
    move_count_since_verify: int = 0


@dataclass
class AnchorPoint:
    """A known, visually identifiable screen location for re-zeroing."""
    name: str
    screen_x: int
    screen_y: int
    template_name: str       # CV template to verify arrival
    tolerance_px: int = 5


@dataclass
class MapperConfig:
    # Confidence decay per unverified move (multiplicative)
    confidence_decay_per_move: float = 0.97
    # Below this confidence, trigger a drift warning
    drift_warn_threshold: float = 0.70
    # Below this confidence, force a re-zero before next command
    desync_threshold: float = 0.45
    # Max moves allowed without a visual verification
    max_unverified_moves: int = 80
    # Re-zero timeout (seconds)
    rezero_timeout_sec: float = 8.0
    # Corner re-zero: how far to overshoot into the corner to guarantee (0,0)
    corner_overshoot_px: int = 3000
    # How often (seconds) to proactively verify position even when "synced"
    periodic_verify_interval_sec: float = 30.0


class CoordinateMapper:
    """
    Bridges absolute screen coordinates (from VisionEngine) to relative
    deltas (for SerialController), maintaining cursor state continuity.
    """

    def __init__(
        self,
        config: MapperConfig | None = None,
        anchors: list[AnchorPoint] | None = None,
    ):
        self.cfg = config or MapperConfig()
        self.anchors = anchors or []
        self._state = CursorState()
        self._sync = SyncState.DESYNCED  # Start desynced — force initial re-zero
        self._lock_callbacks: list[Callable] = []

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[int, int]:
        return (self._state.x, self._state.y)

    @property
    def confidence(self) -> float:
        return self._state.confidence

    @property
    def sync_state(self) -> SyncState:
        return self._sync

    @property
    def needs_rezero(self) -> bool:
        if self._sync == SyncState.DESYNCED:
            return True
        if self._state.confidence < self.cfg.desync_threshold:
            return True
        if self._state.move_count_since_verify >= self.cfg.max_unverified_moves:
            return True
        return False

    @property
    def needs_verify(self) -> bool:
        elapsed = time.time() - self._state.last_verified_at
        if elapsed > self.cfg.periodic_verify_interval_sec:
            return True
        if self._state.confidence < self.cfg.drift_warn_threshold:
            return True
        return False

    # ── Delta computation ────────────────────────────────────────────────

    def compute_delta(self, target_x: int, target_y: int) -> tuple[int, int]:
        """
        Compute the (dx, dy) relative movement from LKP to the target.
        Does NOT update LKP — call `record_move` after execution.
        """
        dx = target_x - self._state.x
        dy = target_y - self._state.y
        return (dx, dy)

    def record_move(self, dx: int, dy: int):
        """
        Update LKP after a relative movement was executed.
        Decays confidence since this is unverified dead-reckoning.
        """
        self._state.x += dx
        self._state.y += dy
        self._state.move_count_since_verify += 1
        self._state.confidence *= self.cfg.confidence_decay_per_move

        # State transitions
        if self._state.confidence < self.cfg.desync_threshold:
            if self._sync != SyncState.DESYNCED:
                log.warning(
                    "Coordinate DESYNC detected (confidence=%.3f, moves=%d). "
                    "Re-zero required.",
                    self._state.confidence,
                    self._state.move_count_since_verify,
                )
                self._sync = SyncState.DESYNCED
        elif self._state.confidence < self.cfg.drift_warn_threshold:
            if self._sync == SyncState.SYNCED:
                log.info(
                    "Drift warning (confidence=%.3f). Verification recommended.",
                    self._state.confidence,
                )
                self._sync = SyncState.DRIFTED

    def record_verified_position(self, x: int, y: int, confidence: float = 1.0):
        """
        Called after a successful CV-based cursor position verification.
        Resets confidence and drift counters.
        """
        old_x, old_y = self._state.x, self._state.y
        drift = math.hypot(x - old_x, y - old_y)

        self._state.x = x
        self._state.y = y
        self._state.confidence = confidence
        self._state.last_verified_at = time.time()
        self._state.move_count_since_verify = 0
        self._sync = SyncState.SYNCED

        if drift > 3:
            log.info(
                "Position verified with %.1fpx drift correction (%d,%d) → (%d,%d).",
                drift, old_x, old_y, x, y,
            )

    # ── Re-zeroing ──────────────────────────────────────────────────────

    def generate_rezero_sequence(
        self,
        method: str = "corner",
        anchor: AnchorPoint | None = None,
    ) -> list[dict]:
        """
        Generate a sequence of commands to re-establish a known position.

        Methods:
          "corner"  — Drive far into top-left corner. The physical screen
                      clamps the cursor at (0,0) regardless of overshoot.
                      Then navigate to the first anchor if available.
          "anchor"  — Navigate directly to a known anchor point (requires
                      a roughly-correct LKP for the initial approach).

        Returns a list of command dicts for the orchestrator to execute:
          {"type": "move_relative", "dx": ..., "dy": ...}
          {"type": "verify_anchor", "anchor": AnchorPoint}
          {"type": "set_position", "x": ..., "y": ...}
        """
        commands: list[dict] = []

        if method == "corner":
            # Phase 1: Overshoot into top-left corner.
            # Regardless of where we are, moving -3000, -3000 guarantees (0,0).
            overshoot = self.cfg.corner_overshoot_px
            commands.append({
                "type": "move_relative",
                "dx": -overshoot,
                "dy": -overshoot,
                "note": "Corner re-zero: driving to (0,0)",
            })
            commands.append({
                "type": "set_position",
                "x": 0, "y": 0,
                "note": "LKP reset to screen origin",
            })

            # Phase 2: Navigate to an anchor for visual confirmation.
            target_anchor = anchor or (self.anchors[0] if self.anchors else None)
            if target_anchor:
                commands.append({
                    "type": "move_relative",
                    "dx": target_anchor.screen_x,
                    "dy": target_anchor.screen_y,
                    "note": f"Navigate to anchor '{target_anchor.name}'",
                })
                commands.append({
                    "type": "verify_anchor",
                    "anchor": target_anchor,
                    "note": "Visual confirmation of re-zero",
                })
                commands.append({
                    "type": "set_position",
                    "x": target_anchor.screen_x,
                    "y": target_anchor.screen_y,
                })

        elif method == "anchor" and anchor:
            dx = anchor.screen_x - self._state.x
            dy = anchor.screen_y - self._state.y
            commands.append({
                "type": "move_relative",
                "dx": dx, "dy": dy,
                "note": f"Navigate to anchor '{anchor.name}' from LKP",
            })
            commands.append({
                "type": "verify_anchor",
                "anchor": anchor,
            })
            commands.append({
                "type": "set_position",
                "x": anchor.screen_x,
                "y": anchor.screen_y,
            })

        return commands

    def execute_rezero(
        self,
        move_fn: Callable[[int, int], bool],
        verify_fn: Callable[[AnchorPoint], tuple[bool, int, int]] | None = None,
        method: str = "corner",
    ) -> bool:
        """
        High-level re-zero executor.

        Args:
            move_fn:   Callable(dx, dy) -> bool. Sends relative movement.
            verify_fn: Callable(anchor) -> (success, actual_x, actual_y).
                       Captures a frame, matches the anchor template,
                       and returns the cursor's true position.
            method:    "corner" or "anchor".

        Returns True if re-zero succeeded and LKP is now trustworthy.
        """
        self._sync = SyncState.REZEROING
        log.info("Starting re-zero (method=%s)...", method)

        sequence = self.generate_rezero_sequence(method=method)

        for step in sequence:
            stype = step["type"]

            if stype == "move_relative":
                ok = move_fn(step["dx"], step["dy"])
                if not ok:
                    log.error("Re-zero move failed.")
                    self._sync = SyncState.DESYNCED
                    return False

            elif stype == "set_position":
                self._state.x = step["x"]
                self._state.y = step["y"]
                log.debug("LKP set to (%d, %d).", step["x"], step["y"])

            elif stype == "verify_anchor" and verify_fn:
                anchor = step["anchor"]
                ok, ax, ay = verify_fn(anchor)
                if ok:
                    self.record_verified_position(ax, ay, confidence=1.0)
                    log.info("Re-zero verified at anchor '%s' (%d, %d).", anchor.name, ax, ay)
                else:
                    log.warning("Anchor verification failed for '%s'.", anchor.name)
                    self._sync = SyncState.DESYNCED
                    return False

        if self._sync == SyncState.REZEROING:
            # No anchor was available, but corner clamp is reliable
            self._state.confidence = 0.90  # Slightly less than verified
            self._state.last_verified_at = time.time()
            self._state.move_count_since_verify = 0
            self._sync = SyncState.SYNCED
            log.info("Re-zero complete (corner clamp, no anchor verification).")

        return True

    # ── CV-based cursor locator (optional enhancement) ──────────────────

    @staticmethod
    def locate_cursor_in_frame(
        frame: np.ndarray,
        cursor_template: np.ndarray,
        cursor_mask: np.ndarray | None = None,
        threshold: float = 0.85,
    ) -> tuple[int, int] | None:
        """
        Attempt to find the mouse cursor sprite in a captured frame.
        Useful for drift verification without moving to an anchor.

        Returns (x, y) of cursor hotspot or None if not found.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray_tmpl = cv2.cvtColor(cursor_template, cv2.COLOR_BGR2GRAY) if len(cursor_template.shape) == 3 else cursor_template

        if cursor_mask is not None:
            result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCORR_NORMED, mask=cursor_mask)
        else:
            result = cv2.matchTemplate(gray_frame, gray_tmpl, cv2.TM_CCORR_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            # Return the hotspot (top-left of cursor sprite; adjust if needed)
            return max_loc
        return None

    # ── Diagnostics ──────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        return {
            "x": self._state.x,
            "y": self._state.y,
            "confidence": round(self._state.confidence, 4),
            "sync_state": self._sync.name,
            "moves_since_verify": self._state.move_count_since_verify,
            "sec_since_verify": round(time.time() - self._state.last_verified_at, 1),
        }
