# Repository Guidelines

## Project Structure & Module Organization
- `app.py`: Flask-SocketIO entrypoint and web server.
- `controllers/`: Arduino, voice, robot, and manipulator controllers (`*_controller.py`).
- `routes/`: HTTP and WebSocket endpoints (`api_routes.py`, `websocket_handlers.py`).
- `utils/`: Small helpers (`csv_utils.py`, `port_utils.py`).
- `templates/`: Web UI (`index.html`).
- `models/`: ML assets and saved models (large files, do not modify casually).
- `config.py` and `hardware_config.json`: Hardware, ports, camera IDs.
- `src/`: Arduino code (`main.cpp`) with `platformio.ini` for builds.

## Build, Test, and Development Commands
- Create venv (recommended): `python -m venv .venv && source .venv/bin/activate`.
- Install deps (from README):
  `pip install flask flask-socketio numpy opencv-python mediapipe pyaudio pyserial google-cloud-speech dynamixel-sdk`.
- Run web app: `python app.py` → visit `http://localhost:5000`.
- Terminal demo: `python terminal_app.py` (optional ASCII/CLI interfaces available).
- Arduino build: `pio run` • Upload: `pio run -t upload` (requires PlatformIO and connected board).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indentation.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_CASE`.
- Structure: keep hardware logic in `controllers/`, transport in `routes/`, helpers in `utils/`.
- Prefer type hints and short, single‑purpose functions; add docstrings for public functions.

## Testing Guidelines
- Current: no formal suite; quick checks via `test.py` and running key flows.
- New tests: place under `tests/` as `test_*.py` using `pytest`. Example: `pytest -q`.
- Aim for smoke tests around `routes/` and controller boundaries; mock hardware/serial.

## Commit & Pull Request Guidelines
- Commits: Conventional style (e.g., `feat: add follower arm sync`, `fix(routes): handle missing port`).
- PRs: clear description, linked issues, repro steps, and screenshots/logs of the web UI.
- Include hardware context (OS, ports, board, camera IDs) when relevant.
- Keep diffs focused; update README or comments when behavior changes.

## Security & Configuration Tips
- Do not commit credentials or GCP keys; use `GOOGLE_APPLICATION_CREDENTIALS` env var.
- Keep `config.py`/`hardware_config.json` machine‑local; avoid hardcoding private ports.
- Large model files in `models/` should not be versioned unless necessary.
