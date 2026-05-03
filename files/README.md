# HIL Testing Framework

A modular Hardware-in-the-Loop system for automated visual inspection and input simulation of 2D grid-based UIs, driven by REST API analytics.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MainDashboard (CTk)                          │
│  ┌──────────┐  ┌───────────┐  ┌───────────┐  ┌──────────────┐  │
│  │ Signals  │  │    Log    │  │ Telemetry │  │   Plugins    │  │
│  │   Tab    │  │    Tab    │  │    Tab    │  │  (dynamic)   │  │
│  └────┬─────┘  └───────────┘  └─────┬─────┘  └──────────────┘  │
│       │                             │                            │
│  ┌────▼─────────────────────────────▼─────────────────────────┐  │
│  │                    SQLite Telemetry DB                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────┬───────────────────┬──────────────────────┬────────────────┘
       │                   │                      │
┌──────▼──────┐   ┌───────▼────────┐   ┌─────────▼──────────┐
│  DataPoller  │   │  VisionEngine  │   │ SerialController   │
│  (threaded)  │   │   (OpenCV)     │   │  (ACK/NAK proto)   │
│              │   │                │   │                    │
│  • /latest   │   │  • Masked      │   │  • Bézier paths    │
│  • /5m       │   │    template    │   │  • Perlin jitter   │
│  • /mapping  │   │    matching    │   │  • Gaussian clicks │
│  • Velocity  │   │  • Canny edge  │   │                    │
│  • Nudging   │   │    density     │   │  ┌──────────────┐  │
│              │   │  • Verify loop │   │  │   Arduino     │  │
└──────────────┘   └────────────────┘   │  │   Leonardo    │  │
                                        │  │   (HID USB)   │  │
                                        │  └──────────────┘  │
                                        └────────────────────┘
```

## Directory Structure

```
hil_framework/
├── config.yaml              # All tunable parameters
├── requirements.txt
├── main_dashboard.py        # CTk GUI + orchestration
├── data_poller.py           # REST API ingestion + analytics
├── vision_engine.py         # OpenCV inspection layer
├── serial_controller.py     # Arduino communication + kinematics
├── modules/                 # Plugin directory (auto-scanned)
│   └── Market_Monitor.py    # Example plugin
├── arduino/
│   └── hid_controller.ino   # Leonardo firmware
├── templates/               # CV template images (*.png)
└── logs/                    # SQLite DB + log files
```

## Quick Start

### 1. Python Environment

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Arduino Firmware

1. Open `arduino/hid_controller.ino` in the Arduino IDE.
2. Select **Board → Arduino Leonardo** (or Acebott Leonardo).
3. Upload.
4. Note the serial port (e.g. `COM5` or `/dev/ttyACM0`).
5. Update `config.yaml → serial.port`.

### 3. Template Images

Place item sprite templates in `./templates/` as PNG files:
- `item_4151.png` — the item sprite (BGRA or BGR)
- `item_4151_mask.png` — optional explicit mask (grayscale)

If the PNG has an alpha channel, the mask is auto-extracted.

### 4. Launch

```bash
python main_dashboard.py
```

## Key Concepts

### Velocity & Nudging

The `VelocityAnalyzer` maintains a sliding window of 5-minute price snapshots and computes a least-squares slope (velocity). The `NudgeEngine` then:

- Sets **buy targets** below the SMA by a configurable offset.
- Sets **sell targets** above the SMA.
- Computes a **break-even floor** = `buy × (1 + tax + margin)`.
- Emits `BUY`/`SELL`/`HOLD`/`SKIP` signals based on velocity direction.

### Masked Template Matching

Standard `matchTemplate` fails on UIs with transparency. By passing an alpha-derived mask, we exclude background pixels from the correlation, dramatically improving accuracy on composite/layered interfaces.

### Biological Motion Modeling

Mouse paths use cubic Bézier curves with randomized control points for natural arcs, overlaid with 1D Perlin noise (multi-octave) for the micro-jitter characteristic of human hand movement. Click durations follow a Gaussian distribution.

### Plugin System

Drop any `.py` file in `./modules/`. If it exports a `register(dashboard)` function, the framework calls it at startup with the full dashboard instance — giving plugins access to the poller, serial controller, vision engine, and UI.

## Configuration Reference

See `config.yaml` for all parameters. Key sections:

| Section      | Controls                                          |
|------------- |---------------------------------------------------|
| `system`     | Log level, DB path, poll intervals, module dir    |
| `serial`     | Port, baud, handshake bytes, retry count          |
| `vision`     | Template dir, confidence, Canny thresholds, grid  |
| `kinematics` | Bézier segments, Perlin params, click timing      |
| `strategy`   | Tax rate, nudge %, volume floors, trend window    |
| `assets`     | Per-item ID, name, volume floor, price buffer     |
