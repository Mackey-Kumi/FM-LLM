"""
Assemble the private Kaggle dataset that air-quality kernels attach:
the fmllm package + the OpenAQ CSVs, in kaggle/datasets/accra_aq/ (gitignored).

    python scripts/build_kaggle_bundle.py
    kaggle datasets create  -p kaggle/datasets/accra_aq            # first time
    kaggle datasets version -p kaggle/datasets/accra_aq -m "..."   # after code/data changes
"""

import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "kaggle" / "datasets" / "accra_aq"
DATA = REPO / "data" / "raw" / "openaq_accra"


def main():
    """
    Rebuild kaggle/datasets/accra_aq/ from scratch: a copy of the fmllm package,
    the two OpenAQ CSVs under data/, and the dataset-metadata.json Kaggle needs.
    It only writes local files; uploading is a separate, manual step.
    """
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(REPO / "fmllm", OUT / "fmllm", ignore=shutil.ignore_patterns("__pycache__"))
    (OUT / "data").mkdir(parents=True)
    for name in ("pm25_hourly.csv", "sensors.csv"):
        shutil.copy2(DATA / name, OUT / "data" / name)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "fm-llm-accra-aq",
        "id": "mackeykumi/fm-llm-accra-aq",
        "licenses": [{"name": "other"}],
    }, indent=2))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"bundle ready: {OUT} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
