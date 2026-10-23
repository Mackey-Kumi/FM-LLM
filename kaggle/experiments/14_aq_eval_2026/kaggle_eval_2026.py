"""
FM-LLM on Accra PM2.5: evaluate experiment 12's model (run 2) on data it has never
been checked on, before it goes live in the demo.

Groups (each with its own tables and day-level bootstrap):
  test_ruleA     Harmattan test (Dec 2025 - Feb 2026), re-scored with rule A: the baseline
  post_trained   Mar - Oct 2026, sensors the model trained on: new season, known places
  post_new       Mar - Oct 2026, sensors installed after training: places never seen
  post_all       Mar - Oct 2026, every sensor

Rule A (fmllm.aq.MIN_REAL_FRACTION): at least 75% of the 48 scored hours must be real
readings, and only real hours are scored; a day's warning event needs >= 18 real hours.
The newer sensors skip about a third of all hours, so the strict rule of experiments
12-13 (every scored hour real) would leave them out entirely.

Looked at once, never tuned on: this is held-out evidence, like the test set.
Locally: `--smoke` uses an untrained tiny model on CPU to check the plumbing.
"""

import argparse
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
import torch

from fmllm.aq import EVENTS, POST_TEST, SPLIT, build_period, new_sensor_ids
from fmllm.evaluation import collect_records, print_summary, write_results
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
        2. Build the forecast windows of both periods with rule A (fmllm.aq.build_period).
        3. Score every forecast once, split the rows into the four groups, and write each
           group's tables and bootstrap into its own folder.

    Command-line:
        --smoke: CPU, untrained tiny model, a few batches (the numbers mean nothing).
        --n-boot: bootstrap resamples (default 2000).
        --smoke-batches: with --smoke, how many batches to score (default 3; 0 = all).

    Writes (to /kaggle/working on Kaggle, ./output_smoke locally), one folder per group:
        test_ruleA/, post_trained/, post_new/, post_all/: each holding test_forecasts.csv,
        test_mae_per_sensor.csv, test_results.json, test_bootstrap.csv
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, untrained tiny model, a few batches")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--smoke-batches", type=int, default=3, help="with --smoke: batches to score (0 = all)")
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

    # 2. both periods, rule A
    csv = find_csv()
    periods = {"test": build_period(csv, model.config, SPLIT["val_end"], SPLIT["test_end"]),
               "post": build_period(csv, model.config, POST_TEST["start"], POST_TEST["end"])}
    new = {str(s) for s in new_sensor_ids(periods["post"].raw)}
    for name, p in periods.items():
        print(f"{name}: {len(p.test)} forecasts at {len({j for j, _ in p.test.items})} sensors")

    # 3. score once per period, then split into groups
    max_batches = (args.smoke_batches or None) if args.smoke else None
    rows = {name: collect_records(model, p.test, device, EVENTS, max_batches=max_batches)
            for name, p in periods.items()}
    groups = {
        "test_ruleA": rows["test"],
        "post_trained": rows["post"][~rows["post"]["sensor"].isin(new)],
        "post_new": rows["post"][rows["post"]["sensor"].isin(new)],
        "post_all": rows["post"],
    }
    for name, df in groups.items():
        print(f"\n{'=' * 30} {name} ({df['sensor'].nunique()} sensors, "
              f"{int((df['method'] == 'fm_llm').sum())} forecasts) {'=' * 30}")
        if df.empty:
            print("  no forecasts in this group")
            continue
        (out_dir / name).mkdir(exist_ok=True)
        summary, per_sensor, boot = write_results(df, EVENTS, out_dir / name, n_boot=args.n_boot,
                                                  extra={"group": name, "model": source, "rule": "A"})
        print_summary(summary, per_sensor, boot, EVENTS)


if __name__ == "__main__":
    main()
