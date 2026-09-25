"""
GridForecast - Main Entry Point
Run from the repo root: python -m app.main
"""

import subprocess
import sys
import time
import urllib.request
import webbrowser

from app.inference import CHECKPOINT_DIR, CHECKPOINT_SPECS, PRECOMPUTED_FILE

API_URL = "http://localhost:8000"
UI_URL = "http://localhost:8501"


def check_dependencies():
    """Need either precomputed forecasts or at least one checkpoint for live inference."""
    checkpoints = [f for f, _ in CHECKPOINT_SPECS.values() if (CHECKPOINT_DIR / f).exists()]
    if PRECOMPUTED_FILE.exists():
        print(f"✅ Precomputed forecasts: {PRECOMPUTED_FILE}")
    if checkpoints:
        print(f"✅ Checkpoints for live inference: {', '.join(checkpoints)}")
    if PRECOMPUTED_FILE.exists() or checkpoints:
        return True

    print("❌ Nothing to serve. Either:")
    print(f"  - put precomputed forecasts at {PRECOMPUTED_FILE}")
    print("    (kaggle/experiments/11_app_forecasts, or: python -m app.precompute_forecasts), or")
    print(f"  - put checkpoint .pt files in {CHECKPOINT_DIR} for live inference")
    return False


def wait_for_api(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{API_URL}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def main():
    print("=" * 60)
    print("⚡ GridForecast")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    print("\n🚀 Starting API...")
    api = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.api.main:app",
        "--host", "127.0.0.1", "--port", "8000",
    ])
    ui = None
    try:
        if not wait_for_api():
            print("❌ API did not become healthy; see the log above.")
            return
        print(f"✅ API ready: {API_URL}  (docs: {API_URL}/docs)")
        print(f"🚀 Starting UI: {UI_URL}\nPress Ctrl+C to stop\n")
        ui = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app/ui/main.py",
            "--server.port", "8501", "--server.headless", "true",
        ])
        time.sleep(2)
        try:
            webbrowser.open(UI_URL)
        except Exception:
            pass
        ui.wait()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        for proc in (ui, api):
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
