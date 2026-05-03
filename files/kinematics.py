"""
kinematics.py — Production kinematic engine with Fitts's Law integration
and multi-phase sub-movement simulation.

Replaces the basic `generate_mouse_path` from serial_controller.py with
a biologically-accurate motor control model:

  Phase 1 — Ballistic Primary Movement
    A fast, slightly imprecise Bézier arc toward the target.
    Speed profiled by Fitts's Law: MT = a + b × log₂(D/W + 1).
    Endpoint scatter follows a 2D Gaussian (σ proportional to speed).

  Phase 2 — Corrective Sub-movement(s)
    One or two short, slower "micro-adjustments" that home in on the
    exact target centre. These have tighter control points and lower
    Perlin amplitude — characteristic of the human closed-loop
    correction phase.

  The result is concatenated into a single waypoint list with natural
  timing metadata per segment.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from serial_controller import PerlinNoise1D, Point, cubic_bezier

# ─── Fitts's Law Model ─────────────────────────────────────────────────────

@dataclass
class FittsParams:
    """
    Empirical Fitts's Law coefficients.
    MT (ms) = a + b × log₂(D / W + 1)

    Defaults are reasonable for desktop mouse use.
    Adjust via config for different input devices.
    """
    a_intercept_ms: float = 50.0     # Base reaction / initiation time
    b_slope_ms: float = 150.0        # Difficulty scaling
    min_mt_ms: float = 80.0          # Floor (very short movements)
    max_mt_ms: float = 1200.0        # Ceiling (full-screen traversals)


def fitts_movement_time(
    distance_px: float,
    target_width_px: float,
    params: FittsParams | None = None,
) -> float:
    """
    Compute expected movement time in milliseconds using Fitts's Law.
    Shannon formulation: MT = a + b × log₂(D/W + 1)
    """
    p = params or FittsParams()
    if distance_px <= 0 or target_width_px <= 0:
        return p.min_mt_ms

    index_of_difficulty = math.log2(distance_px / target_width_px + 1.0)
    mt = p.a_intercept_ms + p.b_slope_ms * index_of_difficulty
    return max(p.min_mt_ms, min(p.max_mt_ms, mt))


# ─── Sub-movement Generator ────────────────────────────────────────────────

@dataclass
class SubMovement:
    """One phase of a multi-phase trajectory."""
    waypoints: list[tuple[int, int]]
    duration_ms: float                # Target traversal time for this phase
    phase: str                        # "ballistic" | "corrective"


@dataclass
class KinematicProfile:
    """Full trajectory with timing metadata."""
    phases: list[SubMovement]
    total_duration_ms: float
    fitts_id: float                   # Index of Difficulty for logging

    @property
    def all_waypoints(self) -> list[tuple[int, int]]:
        """Concatenated waypoint list across all phases."""
        result = []
        for phase in self.phases:
            # Skip first point of subsequent phases (it's the last of the prior)
            start_idx = 1 if result else 0
            result.extend(phase.waypoints[start_idx:])
        return result


def generate_refined_path(
    start: tuple[int, int],
    end: tuple[int, int],
    target_width_px: float = 40.0,
    *,
    # Bézier parameters
    primary_segments: int = 50,
    corrective_segments: int = 15,
    # Perlin noise
    primary_perlin_amp: float = 1.8,
    corrective_perlin_amp: float = 0.5,
    perlin_octaves: int = 4,
    perlin_freq: float = 0.06,
    # Ballistic scatter
    endpoint_scatter_factor: float = 0.08,
    # Fitts's Law
    fitts_params: FittsParams | None = None,
    # RNG
    seed: int | None = None,
) -> KinematicProfile:
    """
    Generate a biologically-plausible mouse trajectory with:
      1. A fast ballistic primary movement (Fitts-timed, slight overshoot).
      2. One or two corrective sub-movements to the exact target.

    Args:
        start:              Current cursor position (px).
        end:                Target position (px).
        target_width_px:    Clickable target width — affects Fitts timing
                            and corrective amplitude.
        primary_segments:   Bézier resolution for the ballistic phase.
        corrective_segments: Bézier resolution for each correction.
        primary_perlin_amp: Jitter magnitude for the ballistic phase.
        corrective_perlin_amp: Jitter magnitude for corrections.
        endpoint_scatter_factor: How far the ballistic phase misses (as
                            fraction of target_width).
        fitts_params:       Fitts's Law coefficients.

    Returns:
        KinematicProfile with all phases and timing metadata.
    """
    rng = random.Random(seed)
    noise = PerlinNoise1D(seed=rng.randint(0, 2**31))
    fp = fitts_params or FittsParams()

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)

    if distance < 2:
        # Trivially close — no movement needed
        return KinematicProfile(
            phases=[SubMovement(waypoints=[start, end], duration_ms=0, phase="trivial")],
            total_duration_ms=0,
            fitts_id=0,
        )

    fitts_id = math.log2(distance / target_width_px + 1.0)
    total_mt = fitts_movement_time(distance, target_width_px, fp)

    phases: list[SubMovement] = []

    # ── Phase 1: Ballistic primary movement ──────────────────────────────
    #
    # The ballistic phase covers ~85-95% of the distance at high speed,
    # but lands slightly off-target. The scatter magnitude is proportional
    # to movement speed (speed-accuracy tradeoff).

    # Compute scattered endpoint
    scatter_radius = target_width_px * endpoint_scatter_factor * (1.0 + fitts_id * 0.15)
    scatter_angle = rng.uniform(0, 2 * math.pi)
    ballistic_end = (
        int(end[0] + scatter_radius * math.cos(scatter_angle)),
        int(end[1] + scatter_radius * math.sin(scatter_angle)),
    )

    # Generate the Bézier arc with strong control-point spread
    p0 = Point(*start)
    p3 = Point(*ballistic_end)
    norm_dx = dx / (distance or 1)
    norm_dy = dy / (distance or 1)

    def primary_control(frac: float) -> Point:
        along_x = start[0] + dx * frac
        along_y = start[1] + dy * frac
        perp_x = -norm_dy
        perp_y = norm_dx
        # Wider arcs for longer distances
        offset = rng.gauss(0, distance * 0.18)
        return Point(along_x + perp_x * offset, along_y + perp_y * offset)

    p1 = primary_control(rng.uniform(0.2, 0.4))
    p2 = primary_control(rng.uniform(0.6, 0.8))

    raw_primary = cubic_bezier(p0, p1, p2, p3, primary_segments)

    # Apply Perlin jitter — amplitude tapers at start and end
    primary_pts = []
    for i, pt in enumerate(raw_primary):
        t_norm = i / max(len(raw_primary) - 1, 1)
        # Bell-shaped amplitude envelope: max jitter at midpoint
        envelope = math.sin(t_norm * math.pi) ** 0.5
        jx = noise.layered(i * perlin_freq, octaves=perlin_octaves) * primary_perlin_amp * envelope
        jy = noise.layered(i * perlin_freq + 73.7, octaves=perlin_octaves) * primary_perlin_amp * envelope
        primary_pts.append((int(round(pt.x + jx)), int(round(pt.y + jy))))

    primary_pts[0] = start
    primary_pts[-1] = ballistic_end

    # Ballistic phase takes ~70-80% of total Fitts time
    ballistic_ratio = rng.uniform(0.70, 0.80)
    ballistic_duration = total_mt * ballistic_ratio

    phases.append(SubMovement(
        waypoints=primary_pts,
        duration_ms=ballistic_duration,
        phase="ballistic",
    ))

    # ── Phase 2: Corrective sub-movement(s) ──────────────────────────────
    #
    # Humans typically make 1-2 corrections. For high-ID movements
    # (small targets, long distance), a second correction is more likely.

    correction_origin = ballistic_end
    remaining_time = total_mt - ballistic_duration

    num_corrections = 1 if fitts_id < 3.5 else (1 + (1 if rng.random() < 0.6 else 0))

    for c_idx in range(num_corrections):
        is_final = (c_idx == num_corrections - 1)
        correction_target = end if is_final else (
            # Intermediate correction: overshoot slightly toward target
            int(end[0] + rng.gauss(0, scatter_radius * 0.3)),
            int(end[1] + rng.gauss(0, scatter_radius * 0.3)),
        )

        c_dx = correction_target[0] - correction_origin[0]
        c_dy = correction_target[1] - correction_origin[1]
        c_dist = math.hypot(c_dx, c_dy)

        if c_dist < 1:
            break

        # Corrections are nearly linear — very tight control points
        cp0 = Point(*correction_origin)
        cp3 = Point(*correction_target)
        # Minimal perpendicular offset for subtle curvature
        c_perp_x = -c_dy / (c_dist or 1)
        c_perp_y = c_dx / (c_dist or 1)
        tight_offset = rng.gauss(0, c_dist * 0.05)
        cp1 = Point(
            correction_origin[0] + c_dx * 0.33 + c_perp_x * tight_offset,
            correction_origin[1] + c_dy * 0.33 + c_perp_y * tight_offset,
        )
        cp2 = Point(
            correction_origin[0] + c_dx * 0.67 + c_perp_x * tight_offset * 0.5,
            correction_origin[1] + c_dy * 0.67 + c_perp_y * tight_offset * 0.5,
        )

        raw_corr = cubic_bezier(cp0, cp1, cp2, cp3, corrective_segments)

        # Much lower jitter for corrective phase
        corr_pts = []
        noise_offset = rng.uniform(200, 500)
        for i, pt in enumerate(raw_corr):
            jx = noise.layered((i + noise_offset) * perlin_freq * 1.5, octaves=2) * corrective_perlin_amp
            jy = noise.layered((i + noise_offset) * perlin_freq * 1.5 + 50, octaves=2) * corrective_perlin_amp
            corr_pts.append((int(round(pt.x + jx)), int(round(pt.y + jy))))

        corr_pts[0] = correction_origin
        if is_final:
            corr_pts[-1] = end  # Lock onto exact target

        # Time allocation: split remaining time across corrections
        # Later corrections are faster (smaller distance)
        corr_time = remaining_time / (num_corrections - c_idx)
        remaining_time -= corr_time

        phases.append(SubMovement(
            waypoints=corr_pts,
            duration_ms=corr_time,
            phase="corrective",
        ))

        correction_origin = correction_target

    # ── Dwell pause between ballistic and corrective ─────────────────────
    # Humans exhibit a brief ~30-80ms "processing pause" before correcting.
    # We inject this by extending the ballistic phase duration slightly.
    dwell_ms = rng.gauss(50, 15)
    phases[0] = SubMovement(
        waypoints=phases[0].waypoints,
        duration_ms=phases[0].duration_ms + max(20, dwell_ms),
        phase=phases[0].phase,
    )

    actual_total = sum(p.duration_ms for p in phases)

    return KinematicProfile(
        phases=phases,
        total_duration_ms=actual_total,
        fitts_id=round(fitts_id, 3),
    )


# ─── Convenience: Per-phase step delay computation ─────────────────────────

def compute_step_delays(profile: KinematicProfile) -> list[float]:
    """
    Convert a KinematicProfile into a flat list of inter-waypoint delays (ms).
    Each delay corresponds to the pause between consecutive waypoints in
    `profile.all_waypoints`.

    The delay within each phase is uniform (duration / num_segments).
    This is simple but effective — the Bézier parameterisation already
    produces non-uniform spatial spacing that creates natural acceleration
    and deceleration.
    """
    delays: list[float] = []
    for phase in profile.phases:
        n = len(phase.waypoints) - 1
        if n <= 0:
            continue
        per_step = phase.duration_ms / n
        # First waypoint of each phase shares a point with the previous phase,
        # so we emit (n) delays to cover (n+1) waypoints.
        # But we skip the shared first point in all_waypoints concatenation,
        # so we need (n) delays for (n) new waypoints.
        delays.extend([per_step] * n)
    return delays


# ─── Speed-adaptive path for SerialController integration ──────────────────

def execute_kinematic_profile(
    profile: KinematicProfile,
    move_fn,
    speed_factor: float = 1.0,
) -> bool:
    """
    Execute a full kinematic profile through a movement function.

    Args:
        profile:      Output from generate_refined_path.
        move_fn:      Callable(dx, dy, delay_ms) -> bool. Sends one relative
                      move and sleeps for `delay_ms`. Return False to abort.
        speed_factor: Global speed multiplier (< 1.0 = faster, > 1.0 = slower).

    Returns True if all moves succeeded.
    """
    import time as _time

    waypoints = profile.all_waypoints
    delays = compute_step_delays(profile)

    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i - 1][0]
        dy = waypoints[i][1] - waypoints[i - 1][1]

        if dx == 0 and dy == 0:
            continue

        delay = delays[i - 1] * speed_factor if i - 1 < len(delays) else 8.0

        ok = move_fn(dx, dy)
        if not ok:
            return False

        _time.sleep(delay / 1000.0)

    return True
