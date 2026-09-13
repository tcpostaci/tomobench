"""Generate and freeze the final 250-target Benchmark v2 corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tomobench.evaluation.benchmark_v2_production import (  # noqa: E402
    PRODUCTION_OUTPUT_DIR,
    ProductionConfig,
    run_production,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PRODUCTION_OUTPUT_DIR)
    parser.add_argument("--base-seed", type=int, default=ProductionConfig().base_seed)
    args = parser.parse_args()
    output = run_production(output_dir=args.output_dir, base_seed=args.base_seed)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
