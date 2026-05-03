"""
orchestrator.py — Full Production Integration.
Handles stochastic sessions, auto-login, and the complete BUY trade cycle.
"""

import logging
import time
import random
import yaml
import mss
import numpy as np
from pathlib import Path

from coordinate_mapper import CoordinateMapper, AnchorPoint, MapperConfig
from data_poller import DataPoller
from serial_controller import SerialController
from vision_engine import VisionEngine, SlotState
from watchdog import (
    Watchdog, WatchdogConfig, HealthAnchor, FaultSignature, 
    RecoveryRoutine, RecoveryStep, HealthState
)

log = logging.getLogger(__name__)

class SessionManager:
    """Handles 'Stochastic Fatigue' for human-mimetic activity patterns."""
    def __init__(self):
        self.is_active = True
        self.next_transition = 0.0
        self.schedule_next_break()

    def schedule_next_break(self):
        # Active duration: 45-180 minutes
        duration = random.uniform(45 * 60, 180 * 60)
        self.next_transition = time.time() + duration
        self.is_active = True
        log.info(f"Session started. Next break in {duration/60:.1f} minutes.")

    def schedule_next_run(self):
        # Break duration: 10-60 minutes
        duration = random.uniform(10 * 60, 60 * 60)
        self.next_transition = time.time() + duration
        self.is_active = False
        log.info(f"Taking a break. Resuming in {duration/60:.1f} minutes.")

    def update(self):
        if time.time() > self.next_transition:
            if self.is_active: self.schedule_next_run()
            else: self.schedule_next_break()

class Orchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)
        
        secrets_path = Path("secrets.yaml")
        self._creds = yaml.safe_load(secrets_path.read_text()) if secrets_path.exists() else None

        self.serial = SerialController(self._cfg)
        self.vision = VisionEngine(self._cfg)
        self.poller = DataPoller(config_path)
	# Temporal nudging (handles the "Stale Trade" logic)
        from temporal_nudge import TemporalNudgeEngine, TTFConfig, EscalationMode
        self.temporal = TemporalNudgeEngine(
            base_engine=self.poller.nudge,
            ttf_config=TTFConfig(stale_threshold_cycles=4, mode=EscalationMode.CAPPED_STEP)
        )
        self.session = SessionManager()
        self.mapper = CoordinateMapper(config=MapperConfig(), anchors=[
            AnchorPoint(name="ge_title", screen_x=320, screen_y=25, template_name="anchor_ge_title")
        ])

        self.watchdog = Watchdog(
            config=WatchdogConfig(
                health_anchors=[HealthAnchor(name="ge_header", template_name="anchor_ge_title", region=(200, 0, 300, 60))],
                fault_signatures=[
                    FaultSignature(name="login_screen", template_name="fault_login", region=(200, 150, 350, 200), indicates=HealthState.DISCONNECTED)
                ],
                recovery_routines=[
                    RecoveryRoutine(name="auto_login", triggers_on=HealthState.DISCONNECTED, steps=[
                        RecoveryStep(action="wait", wait_sec=2.0),
                        RecoveryStep(action="click", target=(425, 280)), # "Existing User"
                        RecoveryStep(action="wait", wait_sec=1.5),
                        RecoveryStep(action="type_creds"),
                        RecoveryStep(action="wait", wait_sec=10.0),
                        RecoveryStep(action="click", target=(400, 300)), # "Play"
                        RecoveryStep(action="verify", verify_anchor="ge_header")
                    ])
                ]
            ),
            vision=self.vision,
            capture_fn=self._capture_frame
        )

    def _capture_frame(self):
        with mss.mss() as sct:
            return np.array(sct.grab(sct.monitors[1]))[:, :, :3]

    def _execute_recovery(self, routine: RecoveryRoutine) -> bool:
        log.info(f"Executing: {routine.name}")
        for step in routine.steps:
            if step.action == "type_creds":
                if not self._creds: 
                    log.error("No secrets.yaml found!"); return False
                self.serial.type_string(self._creds['credentials']['username'])
                self.serial.press_enter()
                time.sleep(random.uniform(0.5, 1.2))
                self.serial.type_string(self._creds['credentials']['password'])
                self.serial.press_enter()
            elif step.action == "click":
                self.click_at(step.target)
            elif step.action == "wait":
                time.sleep(step.wait_sec)
        return True

    def click_at(self, target: tuple[int, int]):
        """Helper to move and click via the kinematic engine."""
        dx, dy = self.mapper.compute_delta(*target)
        # Note: CoordinateMapper and SerialController pathing integration
        # generates the Bézier path internally based on start/end.
        path = self.serial.generate_mouse_path(self.mapper.position, target)
        self.serial.move_to_path(path)
        self.serial.click()
        self.mapper.record_move(dx, dy)

    def step(self, results=None):
        self.session.update()
        if not self.session.is_active:
            log.debug("Session inactive (Break mode). Skipping cycle.")
            return

        if not self.watchdog.is_healthy:
            return

        # Fetch signals from the poller
        temporal_results = self.temporal.evaluate_all(self.poller.assets, self.poller.analyzer)

        for result in temporal_results:
            if result.signal == "BUY":
                log.info(f"Actioning BUY signal: {result.item_name}")
                frame = self._capture_frame()
                if frame is None: continue

                # 1. Locate Empty Slot
                empty_slot = None
                for s in self.vision.slots:
                    insp = self.vision.inspect_slot(frame, s)
                    if insp.state == SlotState.EMPTY:
                        empty_slot = s
                        break

                if empty_slot:
                    center = (empty_slot.x + empty_slot.w // 2, empty_slot.y + empty_slot.h // 2)
                    self.click_at(center)
                    time.sleep(random.uniform(0.7, 1.2))

                    # 2. Search Item
                    self.serial.type_string(result.item_name)
                    time.sleep(random.uniform(1.0, 1.5))

                    # 3. Select Result (Target standard offset for first result)
                    self.click_at((70, 345)) 
                    time.sleep(random.uniform(0.6, 1.0))

                    # 4. Adjust Price (+5% button)
                    self.click_at((275, 215))
                    time.sleep(random.uniform(0.5, 0.8))

                    # 5. Confirm
                    self.click_at((215, 285))
                    log.info(f"Trade confirmed for {result.item_name}")
                    
                    # Prevent multiple trades in one frame to ensure UI stability
                    break 
                else:
                    log.warning("No empty GE slots available.")

    def run(self):
        self.serial.connect()
        self.watchdog.on_recovery_needed(self._execute_recovery)
        self.watchdog.start()
        self.poller.on_update(self.step)
        self.poller.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        self.watchdog.stop(); self.poller.stop(); self.serial.disconnect()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Orchestrator().run()