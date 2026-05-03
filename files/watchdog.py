"""
watchdog.py — Signal-loss detection, obstruction recovery, and system
health monitoring for 24/7 HIL operation.

Runs as a background thread alongside the DataPoller and periodically:
  1. Captures a frame and checks for "healthy" UI anchors.
  2. Detects obstruction states (login screen, connection-lost dialog, etc.)
  3. Executes scripted recovery routines to restore the primary UI.
  4. Gates the DataPoller — pauses signal processing during recovery.
  5. Logs all health transitions to the telemetry DB.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable

import cv2
import numpy as np
import yaml

from vision_engine import VisionEngine, SlotState

log = logging.getLogger(__name__)


# ─── Health States ──────────────────────────────────────────────────────────

class HealthState(Enum):
    HEALTHY = auto()          # Primary UI visible, all anchors present
    DEGRADED = auto()         # Some anchors missing but UI partially functional
    OBSTRUCTED = auto()       # Modal dialog, popup, or overlay blocking UI
    DISCONNECTED = auto()     # "Connection lost" / "Login" screen detected
    UNKNOWN = auto()          # Cannot determine state (capture failure, etc.)
    RECOVERING = auto()       # Recovery routine in progress


# ─── Configuration ──────────────────────────────────────────────────────────

@dataclass
class HealthAnchor:
    """A template that must be present for a given health state."""
    name: str
    template_name: str        # Key in VisionEngine.templates
    region: tuple[int, int, int, int]  # (x, y, w, h) — search region
    required_for: str = "HEALTHY"      # State this anchor validates


@dataclass
class FaultSignature:
    """A template whose presence indicates a specific fault condition."""
    name: str
    template_name: str
    region: tuple[int, int, int, int]
    indicates: HealthState     # What state this template indicates
    confidence: float = 0.80


@dataclass
class RecoveryStep:
    """One action in a recovery sequence."""
    action: str               # "click", "wait", "key", "verify"
    target: tuple[int, int] | None = None  # Click target (absolute px)
    key: str | None = None     # Keypress (for "key" action)
    wait_sec: float = 0.0
    verify_anchor: str | None = None  # Anchor name to verify after action


@dataclass
class RecoveryRoutine:
    """A scripted sequence to recover from a specific fault state."""
    name: str
    triggers_on: HealthState
    steps: list[RecoveryStep]
    max_attempts: int = 3
    cooldown_sec: float = 30.0  # Min time between retry attempts


@dataclass
class WatchdogConfig:
    check_interval_sec: float = 5.0
    max_consecutive_unknowns: int = 6  # After this many, escalate to DISCONNECTED
    health_anchors: list[HealthAnchor] = field(default_factory=list)
    fault_signatures: list[FaultSignature] = field(default_factory=list)
    recovery_routines: list[RecoveryRoutine] = field(default_factory=list)
    db_path: str = "./logs/telemetry.db"


# ─── Watchdog ───────────────────────────────────────────────────────────────

class Watchdog:
    """
    Periodic health monitor that gates the main automation pipeline.

    Integration pattern:
        watchdog = Watchdog(config, vision_engine, capture_fn)
        watchdog.on_state_change(callback)
        watchdog.on_recovery_needed(recovery_executor)
        watchdog.start()

        # In the main loop:
        if watchdog.is_healthy:
            # proceed with normal operations
            ...
    """

    def __init__(
        self,
        config: WatchdogConfig,
        vision: VisionEngine,
        capture_fn: Callable[[], np.ndarray | None],
    ):
        self.cfg = config
        self.vision = vision
        self._capture = capture_fn

        self._state = HealthState.UNKNOWN
        self._prev_state = HealthState.UNKNOWN
        self._consecutive_unknowns = 0
        self._last_healthy_at = 0.0
        self._recovery_attempts: dict[str, int] = {}
        self._recovery_cooldowns: dict[str, float] = {}

        self._state_callbacks: list[Callable[[HealthState, HealthState], None]] = []
        self._recovery_executor: Callable[[RecoveryRoutine], bool] | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()

        self._init_db()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def state(self) -> HealthState:
        with self._lock:
            return self._state

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            return self._state in (HealthState.HEALTHY, HealthState.DEGRADED)

    @property
    def seconds_since_healthy(self) -> float:
        with self._lock:
            if self._state == HealthState.HEALTHY:
                return 0.0
            return time.time() - self._last_healthy_at if self._last_healthy_at > 0 else float("inf")

    # ── Callbacks ────────────────────────────────────────────────────────

    def on_state_change(self, cb: Callable[[HealthState, HealthState], None]):
        """Register callback(old_state, new_state) for health transitions."""
        self._state_callbacks.append(cb)

    def on_recovery_needed(self, executor: Callable[[RecoveryRoutine], bool]):
        """
        Register the recovery executor.
        It receives a RecoveryRoutine and must return True on success.
        The executor is responsible for translating RecoverySteps into
        actual serial commands and CV verifications.
        """
        self._recovery_executor = executor

    # ── Database ─────────────────────────────────────────────────────────

    def _init_db(self):
        con = sqlite3.connect(self.cfg.db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS health_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                ts        REAL    NOT NULL,
                old_state TEXT,
                new_state TEXT,
                detail    TEXT
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS recovery_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                ts         REAL    NOT NULL,
                routine    TEXT,
                attempt    INTEGER,
                success    INTEGER
            )
        """)
        con.commit()
        con.close()

    def _log_transition(self, old: HealthState, new: HealthState, detail: str = ""):
        try:
            con = sqlite3.connect(self.cfg.db_path)
            con.execute(
                "INSERT INTO health_log (ts, old_state, new_state, detail) VALUES (?,?,?,?)",
                (time.time(), old.name, new.name, detail),
            )
            con.commit()
            con.close()
        except Exception as e:
            log.error("Health log write failed: %s", e)

    def _log_recovery(self, routine_name: str, attempt: int, success: bool):
        try:
            con = sqlite3.connect(self.cfg.db_path)
            con.execute(
                "INSERT INTO recovery_log (ts, routine, attempt, success) VALUES (?,?,?,?)",
                (time.time(), routine_name, attempt, 1 if success else 0),
            )
            con.commit()
            con.close()
        except Exception as e:
            log.error("Recovery log write failed: %s", e)

    # ── Health assessment ────────────────────────────────────────────────

    def _assess_health(self, frame: np.ndarray) -> tuple[HealthState, str]:
        """
        Evaluate the current frame against anchors and fault signatures.
        Returns (state, detail_string).
        """
        # Step 1: Check for fault signatures (highest priority)
        for fault in self.cfg.fault_signatures:
            roi = frame[
                fault.region[1] : fault.region[1] + fault.region[3],
                fault.region[0] : fault.region[0] + fault.region[2],
            ]
            match = self.vision.match_template(roi, fault.template_name)
            if match and match.confidence >= fault.confidence:
                return (
                    fault.indicates,
                    f"Fault '{fault.name}' detected (conf={match.confidence:.3f})",
                )

        # Step 2: Check healthy anchors
        anchors_found = 0
        anchors_expected = 0
        missing = []

        for anchor in self.cfg.health_anchors:
            if anchor.required_for != "HEALTHY":
                continue
            anchors_expected += 1
            roi = frame[
                anchor.region[1] : anchor.region[1] + anchor.region[3],
                anchor.region[0] : anchor.region[0] + anchor.region[2],
            ]
            match = self.vision.match_template(roi, anchor.template_name)
            if match and match.matched:
                anchors_found += 1
            else:
                missing.append(anchor.name)

        if anchors_expected == 0:
            # No anchors configured — default to edge density heuristic
            # Check if the frame looks "alive" (has enough visual content)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            edges = cv2.Canny(gray, 50, 150)
            density = float(np.count_nonzero(edges)) / float(edges.size)
            if density > 0.01:
                return HealthState.HEALTHY, f"Edge density OK ({density:.4f})"
            return HealthState.UNKNOWN, f"Low edge density ({density:.4f})"

        if anchors_found == anchors_expected:
            return HealthState.HEALTHY, f"All {anchors_expected} anchors present"
        elif anchors_found > 0:
            return (
                HealthState.DEGRADED,
                f"{anchors_found}/{anchors_expected} anchors (missing: {', '.join(missing)})",
            )
        else:
            return HealthState.OBSTRUCTED, f"No anchors found (missing: {', '.join(missing)})"

    # ── State machine ────────────────────────────────────────────────────

    def _transition(self, new_state: HealthState, detail: str = ""):
        with self._lock:
            old = self._state
            if new_state == old:
                return

            self._prev_state = old
            self._state = new_state

            if new_state == HealthState.HEALTHY:
                self._last_healthy_at = time.time()
                self._consecutive_unknowns = 0

            log.info("Health: %s → %s  (%s)", old.name, new_state.name, detail)
            self._log_transition(old, new_state, detail)

            for cb in self._state_callbacks:
                try:
                    cb(old, new_state)
                except Exception as e:
                    log.error("State change callback error: %s", e)

    # ── Recovery orchestration ───────────────────────────────────────────

    def _attempt_recovery(self):
        """Find and execute the appropriate recovery routine."""
        current = self._state
        if current in (HealthState.HEALTHY, HealthState.DEGRADED, HealthState.RECOVERING):
            return

        if self._recovery_executor is None:
            log.warning("No recovery executor registered. Cannot auto-recover.")
            return

        routine = None
        for r in self.cfg.recovery_routines:
            if r.triggers_on == current:
                routine = r
                break

        if routine is None:
            log.warning("No recovery routine for state %s.", current.name)
            return

        # Check cooldown
        last_attempt_time = self._recovery_cooldowns.get(routine.name, 0.0)
        if time.time() - last_attempt_time < routine.cooldown_sec:
            log.debug("Recovery '%s' in cooldown.", routine.name)
            return

        # Check max attempts
        attempts = self._recovery_attempts.get(routine.name, 0)
        if attempts >= routine.max_attempts:
            log.error(
                "Recovery '%s' exhausted (%d/%d attempts). Manual intervention required.",
                routine.name, attempts, routine.max_attempts,
            )
            return

        # Execute
        self._transition(HealthState.RECOVERING, f"Running '{routine.name}' (attempt {attempts + 1})")
        self._recovery_cooldowns[routine.name] = time.time()
        self._recovery_attempts[routine.name] = attempts + 1

        try:
            success = self._recovery_executor(routine)
        except Exception as e:
            log.error("Recovery executor raised: %s", e)
            success = False

        self._log_recovery(routine.name, attempts + 1, success)

        if success:
            log.info("Recovery '%s' succeeded.", routine.name)
            self._recovery_attempts[routine.name] = 0  # Reset on success
            # Re-assess immediately
            frame = self._capture()
            if frame is not None:
                new_state, detail = self._assess_health(frame)
                self._transition(new_state, detail)
            else:
                self._transition(HealthState.UNKNOWN, "Post-recovery capture failed")
        else:
            log.warning("Recovery '%s' failed (attempt %d).", routine.name, attempts + 1)
            self._transition(current, f"Recovery failed, back to {current.name}")

    # ── Main loop ────────────────────────────────────────────────────────

    def _check_cycle(self):
        frame = self._capture()
        if frame is None:
            self._consecutive_unknowns += 1
            if self._consecutive_unknowns >= self.cfg.max_consecutive_unknowns:
                self._transition(
                    HealthState.DISCONNECTED,
                    f"Capture failed {self._consecutive_unknowns}× consecutively",
                )
            else:
                self._transition(HealthState.UNKNOWN, "Capture returned None")
            return

        self._consecutive_unknowns = 0
        new_state, detail = self._assess_health(frame)
        self._transition(new_state, detail)

        # Trigger recovery if unhealthy
        if not self.is_healthy and self._state != HealthState.RECOVERING:
            self._attempt_recovery()

    def _run(self):
        log.info("Watchdog started (interval=%.1fs).", self.cfg.check_interval_sec)
        while not self._stop.is_set():
            try:
                self._check_cycle()
            except Exception as e:
                log.error("Watchdog cycle error: %s", e)
            self._stop.wait(self.cfg.check_interval_sec)
        log.info("Watchdog stopped.")

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="Watchdog")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    # ── Diagnostics ──────────────────────────────────────────────────────

    def status_dict(self) -> dict:
        with self._lock:
            return {
                "state": self._state.name,
                "previous_state": self._prev_state.name,
                "seconds_since_healthy": round(self.seconds_since_healthy, 1),
                "consecutive_unknowns": self._consecutive_unknowns,
                "recovery_attempts": dict(self._recovery_attempts),
            }
