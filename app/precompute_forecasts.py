"""
Precompute 720h forecasts for a subset of test windows so the demo is
instant and works offline (no Llama, HF token, checkpoints or network).

Run from the repo root (needs the checkpoint .pt files and HF_TOKEN):
    python -m app.precompute_forecasts                      # all available checkpoints, every 24th window
    python -m app.precompute_forecasts --stride 48 --checkpoints instnorm

The Kaggle equivalent (GPU, much faster) is kaggle/experiments/11_app_forecasts;
both write the same file layout (see inference.load_precomputed).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .inference import (
    CHECKPOINT_SPECS,
    PRECOMPUTED_FILE,
    SEQ_LEN,
    ETTh1Data,
    LlamaBackbone,
    load_checkpoint,
    rollout_forecast,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINT_SPECS))
    parser.add_argument("--stride", type=int, default=24, help="Keep every Nth test window (24 = one per day)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=str(PRECOMPUTED_FILE))
    args = parser.parse_args()

    data = ETTh1Data()
    windows = list(range(0, data.num_windows, args.stride))
    print(f"{len(windows)} windows x {len(args.checkpoints)} checkpoints on {args.device}")

    backbone = LlamaBackbone(args.device)
    arrays = {"window_indices": np.array(windows, dtype=np.int64)}

    for name in args.checkpoints:
        embedding_module, decoder, use_instance_norm = load_checkpoint(name, args.device)
        preds = []
        start = time.time()
        for i in range(0, len(windows), args.batch_size):
            batch = windows[i:i + args.batch_size]
            history = torch.from_numpy(
                np.stack([data.test_norm[w:w + SEQ_LEN].T for w in batch])
            ).to(args.device)
            pred = rollout_forecast(history, embedding_module, decoder, backbone, use_instance_norm)
            preds.append(pred.float().cpu().numpy().transpose(0, 2, 1))  # (B, horizon, channels)
            print(f"  {name}: {min(i + args.batch_size, len(windows))}/{len(windows)} "
                  f"({time.time() - start:.0f}s)")
        arrays[name] = np.concatenate(preds).astype(np.float32)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f"✅ Saved {args.out}")


if __name__ == "__main__":
    main()
