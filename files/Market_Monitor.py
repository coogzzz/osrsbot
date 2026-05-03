"""
Market_Monitor — Example plugin (standard tkinter version).
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import scrolledtext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_poller import NudgeResult

log = logging.getLogger(__name__)

_history: list[dict] = []
_text: scrolledtext.ScrolledText | None = None

BG_MID = "#16213e"
FG     = "#e0e0e0"
FG_DIM = "#8a8a9a"


def register(dashboard):
    global _text

    # Add a tab to the notebook
    tab = tk.Frame(dashboard._nb, bg="#1a1a2e")
    dashboard._nb.add(tab, text="  Market Monitor  ")

    tk.Label(
        tab, text="Signal History  (last 100 events)",
        font=("Helvetica Neue", 13, "bold"),
        bg="#1a1a2e", fg=FG, anchor="w",
    ).pack(anchor="w", padx=8, pady=(8, 4))

    _text = scrolledtext.ScrolledText(
        tab, font=("Menlo", 10), bg=BG_MID, fg=FG,
        relief="flat", state="disabled", wrap="none",
    )
    _text.pack(fill="both", expand=True, padx=4, pady=4)

    dashboard.poller.on_update(_on_signals)
    log.info("Market_Monitor plugin initialised.")


def _on_signals(results):
    global _history

    ts = time.strftime("%H:%M:%S")
    for r in results:
        _history.append({
            "time": ts, "item": r.item_name, "signal": r.signal,
            "sma": r.moving_avg, "vel": r.velocity,
            "buy": r.suggested_buy, "sell": r.suggested_sell,
        })

    if len(_history) > 100:
        _history = _history[-100:]

    if _text is not None:
        _text.after(0, _render)


def _render():
    if _text is None:
        return
    _text.configure(state="normal")
    _text.delete("1.0", "end")

    header = f"{'Time':>10s}  {'Item':<24s}  {'Sig':>5s}  {'SMA':>10s}  {'Vel':>7s}  {'Buy':>10s}  {'Sell':>10s}\n"
    _text.insert("end", header)
    _text.insert("end", "─" * 90 + "\n")

    for e in reversed(_history):
        line = (
            f"{e['time']:>10s}  {e['item']:<24s}  {e['signal']:>5s}  "
            f"{e['sma']:>10,.0f}  {e['vel']:>+7.2f}  "
            f"{e['buy']:>10,d}  {e['sell']:>10,d}\n"
        )
        _text.insert("end", line)

    _text.see("1.0")
    _text.configure(state="disabled")
