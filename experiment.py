"""Run the frozen active-versus-random-versus-repeat experiment."""
import argparse
import json
from pathlib import Path
from lens_luotain_target import benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/receipt.json'))
    parser.add_argument('--seeds', type=int, default=64)
    parser.add_argument('--start-seed', type=int, default=6100)
    parser.add_argument('--budget', type=int, default=3)
    args = parser.parse_args()
    record = benchmark(range(args.start_seed, args.start_seed + args.seeds), budget=args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record['summary'], indent=2))


if __name__ == '__main__':
    main()
