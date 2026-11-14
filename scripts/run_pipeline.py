"""
Command-line entry point. Runs the same stage functions the notebook uses.

    python scripts/run_pipeline.py prepare      # manifests, folds, random choices
    python scripts/run_pipeline.py extract      # CNN embeddings (cached, resumable)
    python scripts/run_pipeline.py run          # all Task 1 + Task 2 experiments
    python scripts/run_pipeline.py tables       # summary tables + figures
    python scripts/run_pipeline.py all          # everything in order

Options: --force (rebuild manifests/folds), --models TransferNN,VGGKNN (extract subset)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fvpad import config, pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "extract", "run", "tables", "all"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", default=",".join(config.MODEL_NAMES))
    a = ap.parse_args()
    tables = pipeline.prepare(force=a.force and a.stage in ("prepare", "all"))
    if a.stage in ("extract", "all"):
        pipeline.extract(tables, models=tuple(a.models.split(",")))
    if a.stage in ("run", "all"):
        pipeline.run(tables)
    if a.stage in ("tables", "all"):
        pipeline.tables(tables)


if __name__ == "__main__":
    main()
