"""
GridForecast - Main Entry Point
Run: python -m app.main
"""

import subprocess
import sys
import time
import threading
import webbrowser
from pathlib import Path

def run_api():
    """Run FastAPI server"""
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "app.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])

def run_ui():
    """Run Streamlit UI"""
    # Wait for API to start
    time.sleep(3)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "app/ui/main.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])

def check_dependencies():
    """Check if required files exist"""
    required = [
        Path("kaggle/datasets/checkpoint/etth1_96_instnorm_best.pt"),
        Path("app/cache/backbone_outputs.pkl"),
    ]
    
    missing = [f for f in required if not f.exists()]
    if missing:
        print("❌ Missing required files:")
        for f in missing:
            print(f"  - {f}")
        print("\nTo fix:")
        if not Path("kaggle/datasets/checkpoint/etth1_96_instnorm_best.pt").exists():
            print("  - Checkpoints should be in kaggle/datasets/checkpoint/")
        if not Path("app/cache/backbone_outputs.pkl").exists():
            print("  - Run: python -m app.precache_backbone")
        return False
    return True

def main():
    print("=" * 60)
    print("⚡ GridForecast - Electricity Demand Forecasting")
    print("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting services...")
    print("  - API: http://localhost:8000")
    print("  - UI:  http://localhost:8501")
    print("  - API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    # Start API in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Give API time to start
    time.sleep(3)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass
    
    # Run UI in main thread (blocks)
    try:
        run_ui()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()