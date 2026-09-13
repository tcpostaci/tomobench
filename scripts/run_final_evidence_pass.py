"""Generate the versioned evidence bundle for the final evidence pass."""

from __future__ import annotations

import argparse
from pathlib import Path

from tomobench.evaluation.final_evidence_pass import (
    OPERATOR_RELATIVE,
    PRODUCTION_RELATIVE,
    REVISION_RELATIVE,
    run_final_evidence_pass,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-dir", type=Path, default=PRODUCTION_RELATIVE)
    parser.add_argument("--operator-dir", type=Path, default=OPERATOR_RELATIVE)
    parser.add_argument("--output-dir", type=Path, default=REVISION_RELATIVE)
    args = parser.parse_args()
    result = run_final_evidence_pass(
        production_dir=args.production_dir,
        operator_dir=args.operator_dir,
        output_dir=args.output_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
