"""
SerialController — Hardware-in-the-Loop input layer.
Handles 8-byte packet protocol for Acebott Leonardo (ATmega32U4).
"""

from __future__ import annotations

import logging
import math
import random
import struct
import time
from dataclasses import dataclass

import numpy as np
import serial

log = logging.getLogger(__name__)

# ─── Protocol Constants ──────────────────────────────────────────────────────

CMD_MOVE  = 0x01
CMD_CLICK = 0x02
CMD_PRESS = 0x03
CMD_REL   = 0x04
CMD_PING  = 0x05
CMD_TYPE  = 0x07

RESP_ACK   = 0x06
RESP_NAK   = 0x15
RESP_READY = 0x11

# ─── Motion Jitter (Perlin) ──────────────────────────────────────────────────

class PerlinNoise1D:
    def __init__(self, seed: int | None = None):
        rng = random.Random(seed)
        self._perm = list(range(256))
        rng.shuffle(self._perm)
        self._perm *= 2

    def noise(self, x: float) -> float:
        X = int(math.floor(x)) & 255
        x -= math.floor(x)
        u = x * x * x * (x * (x * 6 - 15) + 10)
        a, b = self._perm[X], self._perm[X + 1]
        grad_a = x if (a & 1) else -x
        grad_b = (x - 1) if (b & 1) else (1 - x)
        return a + u * (grad_b - grad_a)

    def fractal(self, x: float, octaves: int) -> float:
        total, amp, freq, max_val = 0.0, 1.0, 1.0, 0.0
        for _ in range(octaves):
            total += self.noise(x * freq) * amp
            max_val += amp
            amp *= 0.5
            freq *= 2
        return total / max_val

# ─── Trajectory Generation ───────────────────────────────────────────────────

def generate_mouse_path(start: tuple[float, float], end: tuple[float, float], segments: int = 60) -> list[tuple[int, int]]:
    if start == end: return [ (int(start[0]), int(start[1])) ]
    
    dx, dy = end[0] - start[0], end[1] - start[1]
    dist = math.hypot(dx, dy)
    angle = math.atan2(dy, dx)
    
    # Control points for natural arc
    mag = dist * random.uniform(0.2, 0.5)
    p1 = (start[0] + math.cos(angle + 0.4) * mag, start[1] + math.sin(angle + 0.4) * mag)
    p2 = (end[0] - math.cos(angle - 0.4) * mag, end[1] - math.sin(angle - 0.4) * mag)

    path = []
    noise = PerlinNoise1D()
    for i in range(segments + 1):
        t = i / segments
        u = 1 - t
        # Cubic Bezier
        x = (u**3)*start[0] + 3*(u**2)*t*p1[0] + 3*u*(t**2)*p2[0] + (t**3)*end[0]
        y = (u**3)*start[1] + 3*(u**2)*t*p1[1] + 3*u*(t**2)*p2[1] + (t**3)*end[1]
        
        # Jitter
        env = math.sin(t * math.pi)
        jx = noise.fractal(i * 0.1, 3) * 1.5 * env
        jy = noise.fractal(i * 0.1 + 50, 3) * 1.5 * env
        path.append((int(x + jx), int(y + jy)))
        
    return path

# ─── Controller Class ────────────────────────────────────────────────────────

class SerialController:
    def __init__(self, config: dict):
        s_cfg = config.get("serial", {})
        self.port = s_cfg.get("port", "/dev/tty.usbmodemHIDFG1")
        self.baud = s_cfg.get("baud", 115200)
        
        k_cfg = config.get("kinematics", {})
        self.click_mean = k_cfg.get("click_duration_mean_ms", 85)
        self.click_std = k_cfg.get("click_duration_std_ms", 18)
        self.speed_factor = k_cfg.get("movement_speed_factor", 1.0)

        self._ser: serial.Serial | None = None
        self._connected = False

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=2.0)
            time.sleep(2.0) # Bootloader delay
            self._ser.reset_input_buffer()
            if self.ping():
                self._connected = True
                log.info(f"Connected to Arduino on {self.port}")
                return True
            return False
        except Exception as e:
            log.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        if self._ser: self._ser.close()
        self._connected = False

    def _send_packet(self, packet: bytes) -> bool:
        if not self._ser: return False
        try:
            self._ser.write(packet)
            # Check ACK (0x06) then READY (0x11)
            res = self._ser.read(2)
            return len(res) == 2 and res[0] == RESP_ACK and res[1] == RESP_READY
        except Exception as e:
            log.error(f"Serial Error: {e}")
            return False

    # ── Commands ──

    def ping(self) -> bool:
        pkt = struct.pack("<Bhhh", CMD_PING, 0, 0, 0)
        pkt += struct.pack("<B", sum(pkt) & 0xFF)
        return self._send_packet(pkt)

    def move_relative(self, dx: int, dy: int) -> bool:
        pkt = struct.pack("<Bhhh", CMD_MOVE, dx, dy, 0)
        pkt += struct.pack("<B", sum(pkt) & 0xFF)
        return self._send_packet(pkt)

    def click(self, duration_ms: int | None = None) -> bool:
        dur = duration_ms or int(random.gauss(self.click_mean, self.click_std))
        pkt = struct.pack("<Bhhh", CMD_CLICK, 0, 0, max(20, dur))
        pkt += struct.pack("<B", sum(pkt) & 0xFF)
        return self._send_packet(pkt)

    def type_string(self, text: str, wpm: int = 75):
        delay = 60.0 / (wpm * 5)
        for char in text:
            pkt = struct.pack("<Bhhh", CMD_TYPE, ord(char), 0, 0)
            pkt += struct.pack("<B", sum(pkt) & 0xFF)
            self._send_packet(pkt)
            time.sleep(max(0.02, random.gauss(delay, delay * 0.2)))

    def press_enter(self):
        pkt = struct.pack("<Bhhh", CMD_TYPE, 176, 0, 0)
        pkt += struct.pack("<B", sum(pkt) & 0xFF)
        return self._send_packet(pkt)

    def move_to_path(self, path: list[tuple[int, int]]):
        for i in range(1, len(path)):
            dx, dy = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            if dx == 0 and dy == 0: continue
            self.move_relative(dx, dy)
            time.sleep(0.002 * self.speed_factor)