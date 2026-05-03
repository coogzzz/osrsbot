"""
MainDashboard — Standard Tkinter orchestration GUI.

Drop-in replacement using built-in tkinter (no customtkinter dependency).
Works on macOS 10.15+ / any Python 3.8+ without version gates.

Features:
  • Plugin architecture: scans /modules and loads scripts via importlib
  • Real-time scrolling log window (threaded log handler)
  • SQLite telemetry viewer with signal history
  • Live data feed from DataPoller with NudgeResult rendering
  • Serial connection management panel
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sqlite3
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
from typing import Any

import yaml

from data_poller import DataPoller, NudgeResult
from serial_controller import SerialController
from vision_engine import VisionEngine

log = logging.getLogger(__name__)

# ─── Dark theme colours ────────────────────────────────────────────────────

BG       = "#1a1a2e"
BG_MID   = "#16213e"
BG_LIGHT = "#0f3460"
FG       = "#e0e0e0"
FG_DIM   = "#8a8a9a"
ACCENT   = "#3b82f6"
GREEN    = "#22c55e"
RED      = "#ef4444"
YELLOW   = "#eab308"
GREY     = "#6b7280"

FONT       = ("Menlo", 12)
FONT_SM    = ("Menlo", 10)
FONT_TITLE = ("Helvetica Neue", 16, "bold")
FONT_HEAD  = ("Helvetica Neue", 13, "bold")


# ─── Threaded Log Handler → Text widget ────────────────────────────────────

class TextLogHandler(logging.Handler):
    def __init__(self, widget: scrolledtext.ScrolledText, max_lines: int = 2000):
        super().__init__()
        self._w = widget
        self._max = max_lines

    def emit(self, record):
        msg = self.format(record) + "\n"
        try:
            self._w.after(0, self._append, msg)
        except Exception:
            pass

    def _append(self, msg: str):
        self._w.configure(state="normal")
        self._w.insert("end", msg)
        lines = int(self._w.index("end-1c").split(".")[0])
        if lines > self._max:
            self._w.delete("1.0", f"{lines - self._max}.0")
        self._w.see("end")
        self._w.configure(state="disabled")


# ─── Plugin Loader ──────────────────────────────────────────────────────────

class PluginManager:
    def __init__(self, module_dir: str):
        self.module_dir = Path(module_dir)
        self.plugins: dict[str, Any] = {}

    def scan_and_load(self, dashboard):
        if not self.module_dir.exists():
            self.module_dir.mkdir(parents=True, exist_ok=True)
            log.info("Created module directory: %s", self.module_dir)

        for path in sorted(self.module_dir.glob("*.py")):
            if path.name.startswith("_"):
                continue
            name = path.stem
            try:
                spec = importlib.util.spec_from_file_location(name, str(path))
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
                self.plugins[name] = mod

                if hasattr(mod, "register"):
                    try:
                        mod.register(dashboard)
                        log.info("Plugin '%s' registered.", name)
                    except Exception as e:
                        log.warning("Plugin '%s' register() skipped: %s", name, e)
                else:
                    log.info("Plugin '%s' loaded (no register function).", name)
            except Exception as e:
                log.error("Failed to load plugin '%s': %s", name, e)

    def list_plugins(self) -> list[str]:
        return list(self.plugins.keys())


# ─── Telemetry Frame ───────────────────────────────────────────────────────

class TelemetryFrame(tk.Frame):
    def __init__(self, parent, db_path: str, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self._db_path = db_path

        self._text = scrolledtext.ScrolledText(
            self, font=FONT_SM, bg=BG_MID, fg=FG, insertbackground=FG,
            relief="flat", state="disabled", wrap="none",
        )
        self._text.pack(fill="both", expand=True, padx=4, pady=4)

        btn = ttk.Button(
            self, text="Refresh", command=self.refresh,
            style="Blue.TButton",
        )
        btn.pack(pady=(0, 6))

    def refresh(self):
        try:
            con = sqlite3.connect(self._db_path)
            rows = con.execute(
                "SELECT ts, item_id, signal, sma, velocity, buy_target, sell_target, be_floor "
                "FROM signals ORDER BY ts DESC LIMIT 50"
            ).fetchall()
            con.close()
        except Exception as e:
            log.error("Telemetry query failed: %s", e)
            return

        self._text.configure(state="normal")
        self._text.delete("1.0", "end")

        header = f"{'Time':>12s} {'Item':>6s} {'Signal':>6s} {'SMA':>10s} {'Vel':>8s} {'Buy':>10s} {'Sell':>10s} {'BE':>10s}\n"
        self._text.insert("end", header)
        self._text.insert("end", "─" * 80 + "\n")

        for row in rows:
            ts = time.strftime("%H:%M:%S", time.localtime(row[0]))
            line = f"{ts:>12s} {row[1]:>6d} {row[2]:>6s} {row[3]:>10.0f} {row[4]:>8.2f} {row[5]:>10d} {row[6]:>10d} {row[7]:>10d}\n"
            self._text.insert("end", line)

        self._text.configure(state="disabled")


# ─── Signal Card ───────────────────────────────────────────────────────────

class SignalCard(tk.Frame):
    COLORS = {"BUY": GREEN, "SELL": RED, "HOLD": YELLOW, "SKIP": GREY}

    def __init__(self, parent, result: NudgeResult, **kwargs):
        super().__init__(parent, bg=BG_MID, highlightbackground=BG_LIGHT,
                         highlightthickness=1, **kwargs)

        colour = self.COLORS.get(result.signal, GREY)

        top = tk.Frame(self, bg=BG_MID)
        top.pack(fill="x", padx=10, pady=(8, 0))

        tk.Label(
            top, text=result.item_name, font=FONT_HEAD,
            bg=BG_MID, fg=FG, anchor="w",
        ).pack(side="left")

        tk.Label(
            top, text=result.signal, font=("Menlo", 14, "bold"),
            bg=BG_MID, fg=colour, anchor="e",
        ).pack(side="right")

        detail = (
            f"SMA: {result.moving_avg:,.0f}  │  Vel: {result.velocity:+.2f}  │  "
            f"Buy: {result.suggested_buy:,}  Sell: {result.suggested_sell:,}  │  "
            f"BE: {result.break_even_floor:,}  Spread: {result.spread_pct:.1f}%"
        )
        tk.Label(
            self, text=detail, font=FONT_SM,
            bg=BG_MID, fg=FG_DIM, anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 8))


# ─── Main Dashboard ───────────────────────────────────────────────────────

class MainDashboard:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self.root = tk.Tk()
        self.root.title("HIL Framework — Command Dashboard")
        self.root.geometry("1200x780")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        # Configure ttk dark style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_MID, foreground=FG,
                         padding=[14, 6], font=FONT_SM)
        style.map("TNotebook.Tab",
                  background=[("selected", BG_LIGHT)],
                  foreground=[("selected", "white")])
        style.configure("TFrame", background=BG)

        # Button styles that actually render on macOS
        style.configure("Blue.TButton", background=ACCENT, foreground="white",
                         font=FONT_SM, padding=[8, 6])
        style.map("Blue.TButton",
                  background=[("active", BG_LIGHT), ("pressed", BG_LIGHT)])

        style.configure("Green.TButton", background=GREEN, foreground="white",
                         font=FONT_SM, padding=[8, 6])
        style.map("Green.TButton",
                  background=[("active", "#16a34a"), ("pressed", "#16a34a")])

        style.configure("Red.TButton", background=RED, foreground="white",
                         font=FONT_SM, padding=[8, 6])
        style.map("Red.TButton",
                  background=[("active", "#dc2626"), ("pressed", "#dc2626")])

        style.configure("Dark.TButton", background=BG_LIGHT, foreground="white",
                         font=FONT_SM, padding=[8, 6])
        style.map("Dark.TButton",
                  background=[("active", ACCENT), ("pressed", ACCENT)])

        # ── Core subsystems ──
        self.poller = DataPoller(config_path)
        self.serial = SerialController(self._cfg)
        self.vision = VisionEngine(self._cfg)
        self.plugins = PluginManager(self._cfg["system"]["module_scan_dir"])

        self._build_ui()
        self._setup_logging()
        self.plugins.scan_and_load(self)

        self.poller.on_update(self._on_data_update)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._poller_running = False

    # ── UI ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Sidebar ──
        sidebar = tk.Frame(self.root, bg=BG_MID, width=220)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="⬡ HIL Framework", font=FONT_TITLE,
            bg=BG_MID, fg="white",
        ).pack(pady=(20, 10))

        # Serial section
        ser_frame = tk.Frame(sidebar, bg=BG_MID)
        ser_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(ser_frame, text="Serial Port", font=FONT_SM, bg=BG_MID, fg=FG_DIM).pack(anchor="w", padx=4)
        self._port_var = tk.StringVar(value=self._cfg["serial"]["port"])
        self._port_entry = tk.Entry(
            ser_frame, textvariable=self._port_var,
            font=FONT_SM, bg=BG, fg=FG, insertbackground=FG,
            relief="flat", highlightbackground=BG_LIGHT, highlightthickness=1,
        )
        self._port_entry.pack(fill="x", padx=4, pady=2)

        self._connect_btn = ttk.Button(
            ser_frame, text="Connect", command=self._toggle_serial,
            style="Blue.TButton",
        )
        self._connect_btn.pack(fill="x", padx=4, pady=(2, 4))

        self._serial_lbl = tk.Label(
            sidebar, text="● Disconnected", font=FONT_SM, bg=BG_MID, fg=RED,
        )
        self._serial_lbl.pack(pady=2)

        # Poller controls
        self._poll_btn = ttk.Button(
            sidebar, text="▶  Start Poller", command=self._toggle_poller,
            style="Green.TButton",
        )
        self._poll_btn.pack(fill="x", padx=10, pady=5)

        self._poll_once_btn = ttk.Button(
            sidebar, text="Poll Once", command=self._poll_once,
            style="Dark.TButton",
        )
        self._poll_once_btn.pack(fill="x", padx=10, pady=2)

        # Plugin list
        tk.Label(
            sidebar, text="Plugins", font=FONT_HEAD, bg=BG_MID, fg=FG,
        ).pack(anchor="w", padx=12, pady=(14, 2))

        self._plugin_text = scrolledtext.ScrolledText(
            sidebar, height=6, font=FONT_SM, bg=BG, fg=FG_DIM,
            relief="flat", state="disabled", wrap="word",
        )
        self._plugin_text.pack(fill="x", padx=10, pady=2)
        self._refresh_plugin_list()

        # ── Main content (notebook / tabs) ──
        self._nb = ttk.Notebook(self.root)
        self._nb.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Signals tab
        sig_frame = tk.Frame(self._nb, bg=BG)
        self._nb.add(sig_frame, text="  Signals  ")

        sig_canvas = tk.Canvas(sig_frame, bg=BG, highlightthickness=0)
        sig_scroll = tk.Scrollbar(sig_frame, orient="vertical", command=sig_canvas.yview)
        self._signal_inner = tk.Frame(sig_canvas, bg=BG)

        self._signal_inner.bind(
            "<Configure>",
            lambda e: sig_canvas.configure(scrollregion=sig_canvas.bbox("all")),
        )
        sig_canvas.create_window((0, 0), window=self._signal_inner, anchor="nw")
        sig_canvas.configure(yscrollcommand=sig_scroll.set)

        sig_canvas.pack(side="left", fill="both", expand=True)
        sig_scroll.pack(side="right", fill="y")
        self._sig_canvas = sig_canvas

        # Log tab
        log_frame = tk.Frame(self._nb, bg=BG)
        self._nb.add(log_frame, text="  Log  ")

        self._log_text = scrolledtext.ScrolledText(
            log_frame, font=FONT_SM, bg=BG_MID, fg=FG,
            insertbackground=FG, relief="flat", state="disabled", wrap="word",
        )
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Telemetry tab
        tele_frame = tk.Frame(self._nb, bg=BG)
        self._nb.add(tele_frame, text="  Telemetry  ")
        self._telemetry = TelemetryFrame(tele_frame, self._cfg["system"]["db_path"])
        self._telemetry.pack(fill="both", expand=True)

    # ── Logging ──

    def _setup_logging(self):
        handler = TextLogHandler(self._log_text)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s", "%H:%M:%S",
        ))
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        root_logger.setLevel(getattr(logging, self._cfg["system"]["log_level"]))

    # ── Plugin list ──

    def _refresh_plugin_list(self):
        names = self.plugins.list_plugins()
        self._plugin_text.configure(state="normal")
        self._plugin_text.delete("1.0", "end")
        if names:
            self._plugin_text.insert("end", "\n".join(f"  ▸ {n}" for n in names))
        else:
            self._plugin_text.insert("end", "  (none loaded)")
        self._plugin_text.configure(state="disabled")

    # ── Serial ──

    def _toggle_serial(self):
        if self.serial.is_connected:
            self.serial.disconnect()
            self._connect_btn.configure(text="Connect", style="Blue.TButton")
            self._serial_lbl.configure(text="● Disconnected", fg=RED)
        else:
            port = self._port_var.get().strip() or self._cfg["serial"]["port"]
            self.serial.port = port
            ok = self.serial.connect()
            if ok:
                self._connect_btn.configure(text="Disconnect", style="Red.TButton")
                self._serial_lbl.configure(text="● Connected", fg=GREEN)
            else:
                self._serial_lbl.configure(text="● Failed", fg=YELLOW)

    # ── Poller ──

    def _toggle_poller(self):
        if self._poller_running:
            self.poller.stop()
            self._poller_running = False
            self._poll_btn.configure(text="▶  Start Poller", style="Green.TButton")
        else:
            self.poller.start()
            self._poller_running = True
            self._poll_btn.configure(text="■  Stop Poller", style="Red.TButton")

    def _poll_once(self):
        threading.Thread(target=self._do_poll_once, daemon=True).start()

    def _do_poll_once(self):
        results = self.poller.poll_once()
        self._on_data_update(results)

    # ── Data callback ──

    def _on_data_update(self, results: list[NudgeResult]):
        self.root.after(0, self._render_signals, results)

    def _render_signals(self, results: list[NudgeResult]):
        for child in self._signal_inner.winfo_children():
            child.destroy()

        if not results:
            tk.Label(
                self._signal_inner,
                text="No signals yet — waiting for data.",
                font=FONT_SM, bg=BG, fg=FG_DIM,
            ).pack(pady=20)
            return

        for r in results:
            card = SignalCard(self._signal_inner, r)
            card.pack(fill="x", padx=6, pady=3)

        self._refresh_plugin_list()

    # ── Cleanup ──

    def _on_close(self):
        log.info("Shutting down.")
        self.poller.stop()
        self.serial.disconnect()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─── Entry Point ────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO)

    config_path = "config.yaml"
    if not Path(config_path).exists():
        log.error("Config file '%s' not found.", config_path)
        sys.exit(1)

    app = MainDashboard(config_path)
    app.run()


if __name__ == "__main__":
    main()
