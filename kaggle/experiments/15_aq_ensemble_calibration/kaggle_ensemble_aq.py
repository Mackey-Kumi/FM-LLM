"""
FM-LLM on Accra PM2.5: record the dropout ensemble for calibrating the likelihoods.

For every forecast in three periods (rule A windows, fmllm.aq.build_period) it saves
the plain forecast, the 30-member ensemble's hourly median and 10-90% range, every
member's 24-hour averages for "today" and "tomorrow", and the real readings:
  val    Oct - Nov 2025   used to FIT the calibration
  test   Dec 2025 - Feb 2026 (Harmattan)   used to CHECK it
  post   Mar - Oct 2026   used to CHECK it (incl. sensors installed after training)
The calibration itself is done afterwards on CPU from these files (fmllm/calibration.py),
so different methods can be tried without re-running the model.

Locally: `--smoke` uses an untrained tiny model on CPU to check the plumbing.
"""

import argparse
import time
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def package_roots(search_dir):
    """
    Args:
        search_dir: folder to search (recursively).

    Returns:
        List of folders that directly contain an fmllm/ package (the folder you'd
        add to sys.path), i.e. the parent of each fmllm/__init__.py found.
    """
    if not search_dir.exists():
        return []
    return [init.parent.parent for init in search_dir.rglob("fmllm/__init__.py")]


def unpack_zips(inputs, work):
    """
    Unpack fmllm.zip and data.zip from the attached dataset, in case Kaggle kept
    the uploaded folders zipped. Each zip goes into its own folder named after it,
    so the result is work/fmllm/... whether or not the zip stores an fmllm/ prefix.

    Args:
        inputs: /kaggle/input.
        work: folder to unpack into (writable, e.g. /kaggle/working/bundle).
    """
    for z in [*inputs.rglob("fmllm.zip"), *inputs.rglob("data.zip")]:
        with zipfile.ZipFile(z) as f:
            prefixed = all(n.startswith(z.stem + "/") for n in f.namelist())
            f.extractall(work if prefixed else work / z.stem)
        print(f"unpacked {z} -> {work}")


def find_bundle(inputs=Path("/kaggle/input"), work=Path("/kaggle/working/bundle"), local_root=None):
    """
    Locate the folder holding the fmllm package: the attached Kaggle dataset, or
    the local repo when run on this machine. The folder is added to sys.path.

    Args:
        inputs: where Kaggle mounts attached datasets.
        work: where to unpack the bundle zips if Kaggle didn't.
        local_root: the repo root when running locally. Default: three levels up from
            this script's folder (12_.../ -> experiments/ -> kaggle/ -> repo). On Kaggle the
            script runs from /kaggle/src, which has no such folder, so it's skipped.

    Returns:
        Path of the folder that contains fmllm/.

    Raises:
        RuntimeError: if the package can't be found (dataset not attached).
    """
    if local_root is None and len(HERE.parents) > 2:
        local_root = HERE.parents[2]
    roots = package_roots(inputs)
    if not roots and inputs.exists() and next(inputs.rglob("fmllm.zip"), None):
        unpack_zips(inputs, work)
        roots = package_roots(work)
    if local_root is not None and (local_root / "fmllm" / "__init__.py").exists():
        roots.append(local_root)
    if not roots:
        raise RuntimeError(f"fmllm package not found under {inputs} or the local repo: "
                           "is the fm-llm-accra-aq dataset attached?")
    sys.path.insert(0, str(roots[0]))
    return roots[0]


BUNDLE = find_bundle()

# Imported only now: fmllm can be imported only after find_bundle() has put it on sys.path.
# pylint: disable=wrong-import-position,wrong-import-order
import numpy as np
import torch

from fmllm.aq import POST_TEST, SPLIT, build_period
from fmllm.evaluation import collect_ensemble
from fmllm.model import FMLLM, LlamaBackbone, ModelConfig, TinyBackbone, load_checkpoint
# pylint: enable=wrong-import-position,wrong-import-order


def find_csv():
    """
    Locate the PM2.5 data inside the bundle (the folder layout differs between Kaggle and the local repo).

    Returns:
        Path to pm25_hourly.csv.

    Raises:
        FileNotFoundError: if it isn't in any of the expected folders.
    """
    for d in (BUNDLE / "data", BUNDLE / "data" / "raw" / "openaq_accra", BUNDLE):
        if (d / "pm25_hourly.csv").exists():
            return d / "pm25_hourly.csv"
    for top in (Path("/kaggle/input"), Path("/kaggle/working")):  # anywhere in the attached data
        if top.exists():
            hit = next(top.rglob("pm25_hourly.csv"), None)
            if hit:
                return hit
    raise FileNotFoundError("pm25_hourly.csv not found in the bundle")


def find_checkpoint():
    """
    Locate the experiment-12 checkpoint in the attached kernel output.

    Returns:
        Path to aq_pm25_best.pt.

    Raises:
        FileNotFoundError: if experiment 12's output isn't attached (kernel_sources).
    """
    hit = next(Path("/kaggle/input").rglob("aq_pm25_best.pt"), None) if Path("/kaggle/input").exists() else None
    local = HERE.parent / "12_aq_first_train" / "results_run2" / "aq_pm25_best.pt"
    if hit is None and local.exists():
        hit = local
    if hit is None:
        raise FileNotFoundError("aq_pm25_best.pt not found: attach experiment 12's output (kernel_sources)")
    return hit


def main():
    """
    Steps:
        1. Load experiment 12's checkpoint (decoder in bfloat16, as in training), or a tiny
           untrained model with --smoke.
        2. Build the rule-A windows of each period.
        3. Run collect_ensemble on each and save ensemble_<period>.npz.

    Command-line:
        --smoke: CPU, untrained tiny model, a few batches.
        --samples: ensemble size (default 30).
        --smoke-batches: with --smoke, batches per period (default 2; 0 = all).

    Writes (to /kaggle/working on Kaggle, ./output_smoke locally):
        ensemble_val.npz, ensemble_test.npz, ensemble_post.npz (arrays described in
        fmllm.evaluation.collect_ensemble; float arrays stored as float32).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, untrained tiny model, a few batches")
    ap.add_argument("--samples", type=int, default=30, help="ensemble size")
    ap.add_argument("--smoke-batches", type=int, default=2, help="with --smoke: batches per period (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(0)
    out_dir = (Path("/kaggle/working") if Path("/kaggle/working").exists()
               else HERE / ("output_smoke" if args.smoke else "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model
    if args.smoke:
        config = ModelConfig(mlp_dim=32, llm_dim=64, hidden_dim=32)
        device = torch.device("cpu")
        model = FMLLM(config, TinyBackbone(config.llm_dim))
        source = "untrained tiny model (smoke)"
    else:
        assert torch.cuda.is_available(), "No GPU: check the kernel's accelerator setting"
        device = torch.device("cuda:0")
        ckpt = find_checkpoint()
        model = load_checkpoint(ckpt, LlamaBackbone(device), device)
        model.set_decoder_dtype(torch.bfloat16)
        source = str(ckpt)
    print(f"model: {source}")

    # 2-3. each period
    csv = find_csv()
    periods = {"val": (SPLIT["train_end"], SPLIT["val_end"]), "test": (SPLIT["val_end"], SPLIT["test_end"]),
               "post": (POST_TEST["start"], POST_TEST["end"])}
    for name, (start, end) in periods.items():
        t0 = time.time()
        ds = build_period(csv, model.config, start, end).test
        arrays = collect_ensemble(model, ds, device, n_samples=args.samples,
                                  max_batches=(args.smoke_batches or None) if args.smoke else None)
        arrays = {k: (v.astype("float32") if v.dtype.kind == "f" else v) for k, v in arrays.items()}
        np.savez_compressed(out_dir / f"ensemble_{name}.npz", **arrays)
        print(f"{name}: {len(arrays['point'])} forecasts, {len(set(arrays['sensor']))} sensors, "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
