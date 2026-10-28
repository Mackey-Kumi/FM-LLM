"""
Package the forecast service as a Hugging Face Space and upload it.

    python scripts/deploy_space.py                  # stage + upload to MacKey333/accra-air-forecast
    python scripts/deploy_space.py --stage-only     # only build outputs/space/ to inspect it

Free accounts can no longer host Docker or CPU Gradio Spaces, only Gradio Spaces on ZeroGPU
(up to 2). So the Space is a Gradio app on ZeroGPU hardware, but the model runs on the CPU and
never asks for the GPU: no GPU quota is used, by us or by the app calling the API. app.py mounts
a small Gradio status page on the same FastAPI app that serves /v1/* (service/app.py).

The Space gets: the fmllm package, the service, the OpenAQ fetcher, the trained checkpoint,
calibration.json and the downloaded readings (the starting history, and the Harmattan days
replay needs). Llama-3.2-1B is NOT uploaded: the Space downloads it at start-up with the
HF_TOKEN secret, as in training. The secrets HF_TOKEN and OPENAQ_API_KEY are copied from .env;
uploading needs a write token: HF_WRITE_TOKEN in .env, or `hf auth login`.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
from huggingface_hub import HfApi, constants

from fmllm.model import hf_token
from service.live import CALIBRATION, CHECKPOINT, DATA_DIR
# pylint: enable=wrong-import-position

STAGE = REPO / "outputs" / "space"
SPACE = "MacKey333/accra-air-forecast"
HARDWARE = "zero-a10g"         # ZeroGPU: the only free hardware for Gradio Spaces
REQUIREMENTS = """torch==2.13.0
transformers==5.14.1
numpy==2.5.1
pandas==3.0.5
python-dotenv
"""
SPACE_APP = '''"""Space entry point: the forecast API (/v1/*) plus a Gradio status page at /."""
import os

import spaces  # ZeroGPU: imported before torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("SERVICE_DATA_DIR", os.path.join(HERE, "data"))
os.environ.setdefault("SERVICE_CHECKPOINT", os.path.join(HERE, "model", "aq_pm25_best.pt"))
os.environ.setdefault("SERVICE_CALIBRATION", os.path.join(HERE, "model", "calibration.json"))
os.environ.setdefault("SERVICE_WORK_DIR", "/tmp/service")

# pylint: disable=wrong-import-position
import gradio as gr

from service.app import health_body, router, service
# pylint: enable=wrong-import-position


@spaces.GPU
def _unused():
    """ZeroGPU Spaces must declare one GPU function. The model runs on the CPU and never calls this."""


def status():
    """Returns: the /v1/health body, for the status page."""
    return health_body()


with gr.Blocks(title="Accra PM2.5 forecast API") as demo:
    gr.Markdown("# Accra PM2.5 forecast API\\n"
                "FM-LLM forecasts of the next 48 hours for every live OpenAQ sensor in Accra, refreshed hourly.\\n\\n"
                "Endpoints: `/v1/health`, `/v1/sensors`, `/v1/forecasts`, `/v1/forecasts/{sensor_id}`, "
                "`/v1/replay?date=YYYY-MM-DD`")
    box = gr.JSON(label="Service status")
    gr.Button("Refresh").click(status, outputs=box)
    demo.load(status, outputs=box)

service.start()
# Gradio starts the server (ZeroGPU only initialises through demo.launch); then the API routes go
# on Gradio's own app, ahead of Gradio's routes so nothing of Gradio's can shadow /v1/*.
demo.launch(prevent_thread_lock=True, ssr_mode=False)   # SSR puts a Node server in front that hides /v1/*
demo.app.include_router(router)
demo.app.router.routes[:0] = demo.app.router.routes[-len(router.routes):]
del demo.app.router.routes[-len(router.routes):]
demo.block_thread()
'''
README = """---
title: Accra Air Forecast
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.29.1
python_version: "3.12"
app_file: app.py
pinned: false
---

# Accra PM2.5 forecast API

FM-LLM (Fourier embedding + frozen Llama-3.2-1B + FAN-MoE decoder), trained on Accra's OpenAQ sensors,
forecasting the next 48 hours for every live sensor, refreshed hourly. Final-year project, Mackey Junior Kumi.

Endpoints: `/v1/health`, `/v1/sensors`, `/v1/forecasts`, `/v1/forecasts/{sensor_id}`, `/v1/replay?date=YYYY-MM-DD`.
"""


def get_stored_token():
    """Returns: the active token saved by `hf auth login`, or None (not read from HF_TOKEN in .env)."""
    path = Path(constants.HF_TOKEN_PATH)
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def stage():
    """Build outputs/space/ with everything the Space needs."""
    if STAGE.exists():
        shutil.rmtree(STAGE)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO / "fmllm", STAGE / "fmllm", ignore=ignore)
    shutil.copytree(REPO / "service", STAGE / "service", ignore=ignore)
    (STAGE / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "fetch_openaq.py", STAGE / "scripts")
    (STAGE / "model").mkdir()
    shutil.copy(CHECKPOINT, STAGE / "model" / "aq_pm25_best.pt")
    shutil.copy(CALIBRATION, STAGE / "model" / "calibration.json")
    (STAGE / "data").mkdir()
    for name in ("pm25_hourly.csv", "sensors.csv"):
        shutil.copy(DATA_DIR / name, STAGE / "data" / name)
    (STAGE / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (STAGE / "app.py").write_text(SPACE_APP, encoding="utf-8")
    (STAGE / "README.md").write_text(README, encoding="utf-8")
    print(f"staged {sum(1 for _ in STAGE.rglob('*') if _.is_file())} files in {STAGE}")


def upload():
    """
    Create the Space if needed, set its secrets and upload the staged folder.

    Uses HF_WRITE_TOKEN (a write token) to create and upload; the Space itself only gets
    HF_TOKEN (read-only, enough to download Llama), so it never holds a key that can change the account.
    """
    token = hf_token()
    write_token = os.environ.get("HF_WRITE_TOKEN") or get_stored_token()     # .env, or `hf auth login`
    key = os.environ.get("OPENAQ_API_KEY")
    if not token or not write_token or not key:
        raise SystemExit("HF_TOKEN, HF_WRITE_TOKEN and OPENAQ_API_KEY must be in .env")
    api = HfApi(token=write_token)
    api.create_repo(SPACE, repo_type="space", space_sdk="gradio", space_hardware=HARDWARE, exist_ok=True)
    api.add_space_secret(SPACE, "HF_TOKEN", token)
    api.add_space_secret(SPACE, "OPENAQ_API_KEY", key)
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=str(STAGE), commit_message="Deploy forecast service")
    user, name = SPACE.split("/")
    print(f"uploaded: https://huggingface.co/spaces/{SPACE}")
    print(f"API:      https://{user.lower()}-{name}.hf.space/v1/health")


def main():
    """Stage, then upload unless --stage-only."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-only", action="store_true")
    args = ap.parse_args()
    stage()
    if not args.stage_only:
        upload()


if __name__ == "__main__":
    main()
