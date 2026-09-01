from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.ebmc_feature_dataset import (
    build_chsims_feature_cache,
    build_ebmc_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact tensor cache from utterance-level MSA features."
    )
    parser.add_argument(
        "--dataset", choices=["CMUMOSI", "CMUMOSEI", "CHSIMS"], default="CMUMOSEI"
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label-file")
    parser.add_argument("--manifest-file", default="manifest/chsims_manifest.csv")
    parser.add_argument("--limit-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or PROJECT_ROOT / "datasets" / args.dataset
    output = args.output or PROJECT_ROOT / "cache" / f"ebmc_{args.dataset.lower()}_full.pt"
    if args.dataset == "CHSIMS":
        build_chsims_feature_cache(
            dataset_root,
            output,
            manifest_file=args.manifest_file,
            limit_per_split=args.limit_per_split,
        )
    else:
        label_file = args.label_file or f"{args.dataset}_features_raw_2way.pkl"
        build_ebmc_feature_cache(
            dataset_root,
            output,
            label_file=label_file,
            dataset_name=args.dataset,
            limit_per_split=args.limit_per_split,
        )


if __name__ == "__main__":
    main()
