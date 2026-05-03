"""
DataPoller — Multi-threaded data ingestion, velocity analysis, and nudging engine.

Pulls from the OSRS Wiki Prices API and computes:
  • 5-minute moving averages
  • Price velocity (∆price / ∆time over a sliding window)
  • Liquidity scores (volume-weighted)
  • Nudged entry/exit targets with break-even floor accounting for GE tax
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import requests
import yaml

log = logging.getLogger(__name__)

# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class PriceSnapshot:
    item_id: int
    timestamp: float
    high: int
    low: int
    high_volume: int
    low_volume: int

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def spread(self) -> float:
        if self.mid == 0:
            return 0.0
        return abs(self.high - self.low) / self.mid * 100.0

    @property
    def total_volume(self) -> int:
        return self.high_volume + self.low_volume


@dataclass
class NudgeResult:
    item_id: int
    item_name: str
    moving_avg: float
    velocity: float              # GP per 5-min interval
    liquidity_score: float
    suggested_buy: int
    suggested_sell: int
    break_even_floor: int
    spread_pct: float
    signal: str                  # "BUY" | "SELL" | "HOLD" | "SKIP"


@dataclass
class AssetConfig:
    id: int
    name: str
    volume_floor: int
    price_buffer_pct: float


# ─── Mapping Cache ──────────────────────────────────────────────────────────

class ItemMapping:
    """Loads the /mapping endpoint once and provides id↔name lookups."""

    def __init__(self, base_url: str, user_agent: str):
        self._by_id: dict[int, dict] = {}
        self._load(base_url, user_agent)

    def _load(self, base_url: str, ua: str):
        try:
            resp = requests.get(
                f"{base_url}/mapping",
                headers={"User-Agent": ua},
                timeout=10,
            )
            resp.raise_for_status()
            for entry in resp.json():
                self._by_id[entry["id"]] = entry
            log.info("Loaded %d item mappings.", len(self._by_id))
        except Exception as e:
            log.error("Failed to load item mapping: %s", e)

    def name(self, item_id: int) -> str:
        return self._by_id.get(item_id, {}).get("name", f"Unknown({item_id})")

    def members(self, item_id: int) -> bool:
        return self._by_id.get(item_id, {}).get("members", False)


# ─── Velocity & Liquidity Analyzer ─────────────────────────────────────────

class VelocityAnalyzer:
    """
    Maintains a sliding window of PriceSnapshots per item and computes:
      • Simple Moving Average (SMA) of mid prices
      • Price velocity (linear slope over the window)
      • Liquidity score (normalised volume × inverse spread)
    """

    def __init__(self, window_size: int = 6):
        self._window_size = window_size
        self._buffers: dict[int, deque[PriceSnapshot]] = {}

    def ingest(self, snap: PriceSnapshot):
        buf = self._buffers.setdefault(
            snap.item_id, deque(maxlen=self._window_size)
        )
        buf.append(snap)

    def sma(self, item_id: int) -> float | None:
        buf = self._buffers.get(item_id)
        if not buf:
            return None
        return sum(s.mid for s in buf) / len(buf)

    def velocity(self, item_id: int) -> float | None:
        """
        Returns the average ∆mid per interval using least-squares slope.
        Positive = price rising, negative = falling.
        """
        buf = self._buffers.get(item_id)
        if not buf or len(buf) < 2:
            return None
        n = len(buf)
        xs = list(range(n))
        ys = [s.mid for s in buf]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def liquidity_score(self, item_id: int) -> float | None:
        """
        Composite score: higher volume + tighter spread = more liquid.
        Score = avg_volume / (1 + avg_spread_pct)
        """
        buf = self._buffers.get(item_id)
        if not buf:
            return None
        avg_vol = sum(s.total_volume for s in buf) / len(buf)
        avg_spread = sum(s.spread for s in buf) / len(buf)
        return avg_vol / (1.0 + avg_spread)

    def has_sufficient_data(self, item_id: int) -> bool:
        buf = self._buffers.get(item_id)
        return buf is not None and len(buf) >= 2


# ─── Nudging Algorithm ─────────────────────────────────────────────────────

class NudgeEngine:
    """
    Computes target prices:
      • Buy  = SMA × (1 - nudge_offset)
      • Sell = SMA × (1 + nudge_offset)
      • Break-even floor = buy_price × (1 + tax + margin)

    Signal logic:
      • SKIP   if spread too wide or volume too low
      • BUY    if velocity < 0 (falling — accumulate)
      • SELL   if velocity > 0 (rising — distribute)
      • HOLD   otherwise
    """

    def __init__(self, cfg: dict):
        self.tax_pct = cfg["transaction_tax_pct"] / 100.0
        self.nudge_pct = cfg["nudge_offset_pct"] / 100.0
        self.margin_pct = cfg["break_even_margin_pct"] / 100.0
        self.min_vol = cfg["min_volume_5m"]
        self.max_spread = cfg["max_spread_pct"]

    def evaluate(
        self,
        asset: AssetConfig,
        analyzer: VelocityAnalyzer,
    ) -> NudgeResult | None:
        if not analyzer.has_sufficient_data(asset.id):
            return None

        sma = analyzer.sma(asset.id)
        vel = analyzer.velocity(asset.id)
        liq = analyzer.liquidity_score(asset.id)
        if sma is None or vel is None or liq is None:
            return None

        # Spread / volume filter
        buf = analyzer._buffers[asset.id]
        latest = buf[-1]
        spread_pct = latest.spread
        if spread_pct > self.max_spread or latest.total_volume < self.min_vol:
            return NudgeResult(
                item_id=asset.id, item_name=asset.name,
                moving_avg=sma, velocity=vel, liquidity_score=liq,
                suggested_buy=0, suggested_sell=0, break_even_floor=0,
                spread_pct=spread_pct, signal="SKIP",
            )

        # Apply per-asset buffer on top of global nudge
        effective_nudge = self.nudge_pct + (asset.price_buffer_pct / 100.0)

        buy_target = int(sma * (1.0 - effective_nudge))
        sell_target = int(sma * (1.0 + effective_nudge))
        be_floor = int(buy_target * (1.0 + self.tax_pct + self.margin_pct))

        # Ensure sell > break-even
        if sell_target < be_floor:
            sell_target = be_floor + 1

        # Directional signal
        if vel < -0.5:
            signal = "BUY"
        elif vel > 0.5:
            signal = "SELL"
        else:
            signal = "HOLD"

        return NudgeResult(
            item_id=asset.id, item_name=asset.name,
            moving_avg=round(sma, 1), velocity=round(vel, 2),
            liquidity_score=round(liq, 2),
            suggested_buy=buy_target, suggested_sell=sell_target,
            break_even_floor=be_floor, spread_pct=round(spread_pct, 2),
            signal=signal,
        )


# ─── REST Poller (threaded) ────────────────────────────────────────────────

class DataPoller:
    """
    Continuously polls the OSRS Wiki Prices API on a background thread.
    Feeds snapshots into the VelocityAnalyzer, runs NudgeEngine evaluations,
    and pushes results to registered callbacks.
    """

    BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
    USER_AGENT = "HIL-Framework/1.0 (research)"

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        strat = self._cfg["strategy"]
        sys_cfg = self._cfg["system"]

        self.poll_interval = sys_cfg["poll_interval_sec"]
        self.assets = [AssetConfig(**a) for a in self._cfg["assets"]]
        self.asset_ids = {a.id for a in self.assets}

        self.mapping = ItemMapping(self.BASE_URL, self.USER_AGENT)
        self.analyzer = VelocityAnalyzer(window_size=strat["trend_window"])
        self.nudge = NudgeEngine(strat)

        self._callbacks: list[Callable[[list[NudgeResult]], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Telemetry DB
        self._db_path = sys_cfg["db_path"]
        self._init_db()

    # ── Database ──

    def _init_db(self):
        con = sqlite3.connect(self._db_path)
        con.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id   INTEGER NOT NULL,
                ts        REAL    NOT NULL,
                high      INTEGER,
                low       INTEGER,
                high_vol  INTEGER,
                low_vol   INTEGER
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                ts            REAL    NOT NULL,
                item_id       INTEGER NOT NULL,
                signal        TEXT,
                sma           REAL,
                velocity      REAL,
                buy_target    INTEGER,
                sell_target   INTEGER,
                be_floor      INTEGER
            )
        """)
        con.commit()
        con.close()

    def _store_snapshot(self, snap: PriceSnapshot):
        con = sqlite3.connect(self._db_path)
        con.execute(
            "INSERT INTO snapshots (item_id,ts,high,low,high_vol,low_vol) VALUES (?,?,?,?,?,?)",
            (snap.item_id, snap.timestamp, snap.high, snap.low,
             snap.high_volume, snap.low_volume),
        )
        con.commit()
        con.close()

    def _store_signal(self, r: NudgeResult):
        con = sqlite3.connect(self._db_path)
        con.execute(
            "INSERT INTO signals (ts,item_id,signal,sma,velocity,buy_target,sell_target,be_floor) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (time.time(), r.item_id, r.signal, r.moving_avg, r.velocity,
             r.suggested_buy, r.suggested_sell, r.break_even_floor),
        )
        con.commit()
        con.close()

    # ── Callbacks ──

    def on_update(self, cb: Callable[[list[NudgeResult]], None]):
        self._callbacks.append(cb)

    # ── Polling ──

    def _fetch_5m(self) -> dict:
        """Fetch the /5m endpoint — returns 5-minute averaged prices."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/5m",
                headers={"User-Agent": self.USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("data", {})
        except Exception as e:
            log.error("5m fetch failed: %s", e)
            return {}

    def _fetch_latest(self) -> dict:
        """Fetch /latest for real-time instant prices."""
        try:
            resp = requests.get(
                f"{self.BASE_URL}/latest",
                headers={"User-Agent": self.USER_AGENT},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("data", {})
        except Exception as e:
            log.error("latest fetch failed: %s", e)
            return {}

    def _poll_cycle(self):
        now = time.time()
        data_5m = self._fetch_5m()
        data_latest = self._fetch_latest()

        for asset in self.assets:
            sid = str(asset.id)

            # Prefer 5m averages; fall back to /latest
            entry = data_5m.get(sid) or data_latest.get(sid)
            if not entry:
                continue

            snap = PriceSnapshot(
                item_id=asset.id,
                timestamp=now,
                high=entry.get("avgHighPrice") or entry.get("high", 0) or 0,
                low=entry.get("avgLowPrice") or entry.get("low", 0) or 0,
                high_volume=entry.get("highPriceVolume", 0) or 0,
                low_volume=entry.get("lowPriceVolume", 0) or 0,
            )

            with self._lock:
                self.analyzer.ingest(snap)
            self._store_snapshot(snap)

        # Evaluate all assets
        results: list[NudgeResult] = []
        for asset in self.assets:
            with self._lock:
                result = self.nudge.evaluate(asset, self.analyzer)
            if result:
                results.append(result)
                self._store_signal(result)
                log.info(
                    "[%s] sig=%s  sma=%.0f  vel=%.2f  buy=%d  sell=%d  BE=%d",
                    result.item_name, result.signal, result.moving_avg,
                    result.velocity, result.suggested_buy,
                    result.suggested_sell, result.break_even_floor,
                )

        for cb in self._callbacks:
            try:
                cb(results)
            except Exception as e:
                log.error("Callback error: %s", e)

    def _run(self):
        log.info("DataPoller thread started (interval=%ds).", self.poll_interval)
        while not self._stop.is_set():
            try:
                self._poll_cycle()
            except Exception as e:
                log.error("Poll cycle error: %s", e)
            self._stop.wait(self.poll_interval)
        log.info("DataPoller thread stopped.")

    # ── Lifecycle ──

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="DataPoller")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def poll_once(self) -> list[NudgeResult]:
        """Synchronous single poll for testing / manual trigger."""
        self._poll_cycle()
        results = []
        for asset in self.assets:
            with self._lock:
                r = self.nudge.evaluate(asset, self.analyzer)
            if r:
                results.append(r)
        return results
