"""
VisionEngine — Automated Visual Inspection (AVI) of a 2D grid-based UI.

Features:
  • Masked template matching (cv2.matchTemplate with TM_CCORR_NORMED + mask)
    to handle dynamic transparency and background noise.
  • Canny edge-density classification for Empty vs Occupied slots.
  • Post-input verification loop that confirms UI state transitions.
  • Thread-safe screen capture with configurable ROI grid.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)


class SlotState(Enum):
    EMPTY = auto()
    OCCUPIED = auto()
    UNKNOWN = auto()


@dataclass
class SlotROI:
    """Bounding box for a single UI slot."""
    index: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class MatchResult:
    """Result of a template match against a single slot."""
    slot: SlotROI
    confidence: float
    location: tuple[int, int]      # (x, y) of best match within the slot
    matched: bool


@dataclass
class SlotInspection:
    """Full inspection result for one slot."""
    slot: SlotROI
    state: SlotState
    edge_density: float
    match: MatchResult | None


# ─── Template Cache ─────────────────────────────────────────────────────────

class TemplateStore:
    """
    Loads template images + optional alpha masks from a directory.
    Expected structure:
        templates/
          item_4151.png        (BGRA or BGR)
          item_4151_mask.png   (optional explicit grayscale mask)
    If the template has an alpha channel, it's auto-extracted as the mask.
    """

    def __init__(self, template_dir: str):
        self._templates: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._load_all(Path(template_dir))

    def _load_all(self, d: Path):
        if not d.exists():
            log.warning("Template directory %s does not exist.", d)
            return
        for p in sorted(d.glob("*.png")):
            if p.stem.endswith("_mask"):
                continue  # handled with parent
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            mask = None
            mask_path = p.with_name(p.stem + "_mask.png")
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            elif img.shape[2] == 4:
                # Extract alpha channel as mask
                mask = img[:, :, 3]
                img = img[:, :, :3]

            self._templates[p.stem] = (img, mask)
            log.debug("Loaded template '%s' (%s mask).", p.stem, "with" if mask is not None else "no")

        log.info("Loaded %d templates from %s.", len(self._templates), d)

    def get(self, name: str) -> tuple[np.ndarray, np.ndarray | None] | None:
        return self._templates.get(name)

    def all(self) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
        return dict(self._templates)


# ─── Core Vision Engine ─────────────────────────────────────────────────────

class VisionEngine:
    """
    Performs slot-based inspection on a captured frame.
    Designed for a static 2D grid UI with dynamic transparency.
    """

    def __init__(self, config: dict):
        vis_cfg = config["vision"]
        self.confidence_threshold = vis_cfg["confidence_threshold"]
        self.canny_low = vis_cfg["canny_low"]
        self.canny_high = vis_cfg["canny_high"]
        self.empty_edge_max = vis_cfg["empty_edge_density_max"]
        self.verify_timeout = vis_cfg["verification_timeout_sec"]
        self.verify_poll_ms = vis_cfg["verification_poll_ms"]

        self.slots = [
            SlotROI(index=i, x=b[0], y=b[1], w=b[2], h=b[3])
            for i, b in enumerate(vis_cfg["slot_grid"])
        ]

        self.templates = TemplateStore(vis_cfg["template_dir"])

    # ── Frame capture (stub — replace with actual capture source) ──

    @staticmethod
    def capture_frame(source=None) -> np.ndarray | None:
        """
        Capture a frame from the inspection target.
        In production, replace with:
          - HDMI capture card (cv2.VideoCapture)
          - NDI stream
          - Shared memory / virtual display

        For development, accepts a file path or ndarray directly.
        """
        if source is None:
            log.error("No capture source provided.")
            return None
        if isinstance(source, np.ndarray):
            return source
        if isinstance(source, (str, Path)):
            frame = cv2.imread(str(source))
            if frame is None:
                log.error("Could not read frame from %s", source)
            return frame
        return None

    # ── Slot extraction ──

    def extract_slot(self, frame: np.ndarray, slot: SlotROI) -> np.ndarray:
        return frame[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w].copy()

    # ── Masked template matching ──

    def match_template(
        self,
        roi: np.ndarray,
        template_name: str,
    ) -> MatchResult | None:
        """
        Runs cv2.matchTemplate with an optional mask (TM_CCORR_NORMED).
        The mask allows ignoring transparent / background pixels.
        """
        entry = self.templates.get(template_name)
        if entry is None:
            return None

        tmpl, mask = entry

        # Resize template if it's larger than the ROI
        if tmpl.shape[0] > roi.shape[0] or tmpl.shape[1] > roi.shape[1]:
            scale = min(
                roi.shape[0] / tmpl.shape[0],
                roi.shape[1] / tmpl.shape[1],
            ) * 0.9
            new_w = int(tmpl.shape[1] * scale)
            new_h = int(tmpl.shape[0] * scale)
            tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if mask is not None:
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Ensure both are the same colour depth
        if len(roi.shape) == 2:
            roi_gray = roi
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if len(tmpl.shape) == 3 else tmpl
        else:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY) if len(tmpl.shape) == 3 else tmpl

        method = cv2.TM_CCORR_NORMED

        if mask is not None:
            # Ensure mask is single-channel, same size as template
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(roi_gray, tmpl_gray, method, mask=mask)
        else:
            result = cv2.matchTemplate(roi_gray, tmpl_gray, method)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        return MatchResult(
            slot=SlotROI(0, 0, 0, 0, 0),  # caller fills in
            confidence=float(max_val),
            location=max_loc,
            matched=max_val >= self.confidence_threshold,
        )

    # ── Canny edge-density slot classification ──

    def classify_slot_state(self, roi: np.ndarray) -> tuple[SlotState, float]:
        """
        Uses Canny edge detection to determine if a slot is empty or occupied.
        An empty slot will have very few edges (just UI chrome).
        An occupied slot will have significant edge activity from the item sprite.
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        density = float(np.count_nonzero(edges)) / float(edges.size)

        if density <= self.empty_edge_max:
            return SlotState.EMPTY, density
        return SlotState.OCCUPIED, density

    # ── Full slot inspection ──

    def inspect_slot(
        self,
        frame: np.ndarray,
        slot: SlotROI,
        template_name: str | None = None,
    ) -> SlotInspection:
        roi = self.extract_slot(frame, slot)
        state, density = self.classify_slot_state(roi)

        match = None
        if template_name and state == SlotState.OCCUPIED:
            match = self.match_template(roi, template_name)
            if match:
                match.slot = slot

        return SlotInspection(
            slot=slot,
            state=state,
            edge_density=round(density, 5),
            match=match,
        )

    def inspect_all_slots(
        self,
        frame: np.ndarray,
        template_name: str | None = None,
    ) -> list[SlotInspection]:
        return [self.inspect_slot(frame, s, template_name) for s in self.slots]

    # ── Post-input verification ──

    def verify_state_change(
        self,
        capture_fn,
        slot: SlotROI,
        expected_state: SlotState,
        template_name: str | None = None,
    ) -> bool:
        """
        Polls the slot until its state matches `expected_state` or timeout.
        `capture_fn` should return a fresh np.ndarray frame on each call.

        Returns True if the expected state was observed within the timeout.
        """
        deadline = time.time() + self.verify_timeout
        poll_sec = self.verify_poll_ms / 1000.0

        while time.time() < deadline:
            frame = capture_fn()
            if frame is None:
                time.sleep(poll_sec)
                continue

            inspection = self.inspect_slot(frame, slot, template_name)

            if inspection.state == expected_state:
                log.info(
                    "Verification OK: slot %d is %s (density=%.5f).",
                    slot.index, expected_state.name, inspection.edge_density,
                )
                return True

            time.sleep(poll_sec)

        log.warning(
            "Verification TIMEOUT: slot %d did not reach %s within %.1fs.",
            slot.index, expected_state.name, self.verify_timeout,
        )
        return False

    def find_empty_slot(self, frame: np.ndarray) -> SlotROI | None:
        """Return the first empty slot, or None."""
        for slot in self.slots:
            state, _ = self.classify_slot_state(self.extract_slot(frame, slot))
            if state == SlotState.EMPTY:
                return slot
        return None

    def find_matching_slot(
        self,
        frame: np.ndarray,
        template_name: str,
    ) -> SlotInspection | None:
        """Find the slot that best matches a template above the confidence threshold."""
        best: SlotInspection | None = None
        best_conf = 0.0
        for slot in self.slots:
            insp = self.inspect_slot(frame, slot, template_name)
            if insp.match and insp.match.matched and insp.match.confidence > best_conf:
                best = insp
                best_conf = insp.match.confidence
        return best

    # ── Debug visualization ──

    def annotate_frame(
        self,
        frame: np.ndarray,
        inspections: Sequence[SlotInspection],
    ) -> np.ndarray:
        """Draw slot bounding boxes and state labels on a frame copy for debugging."""
        vis = frame.copy()
        colours = {
            SlotState.EMPTY: (0, 255, 0),
            SlotState.OCCUPIED: (0, 165, 255),
            SlotState.UNKNOWN: (0, 0, 255),
        }
        for insp in inspections:
            s = insp.slot
            c = colours.get(insp.state, (128, 128, 128))
            cv2.rectangle(vis, (s.x, s.y), (s.x + s.w, s.y + s.h), c, 2)
            label = f"S{s.index} {insp.state.name} ({insp.edge_density:.4f})"
            cv2.putText(
                vis, label, (s.x, s.y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1, cv2.LINE_AA,
            )
            if insp.match and insp.match.matched:
                conf_label = f"{insp.match.confidence:.2f}"
                cv2.putText(
                    vis, conf_label,
                    (s.x + s.w - 50, s.y + s.h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA,
                )
        return vis
