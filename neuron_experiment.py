"""Reproduce the frozen memory-gate-soma test and emit a compact receipt."""
import argparse
import json
from pathlib import Path

from memory_gate_soma import MemorySetup, benchmark


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start-seed', type=int, default=9100)
    p.add_argument('--seeds', type=int, default=512)
    p.add_argument('--budget', type=int, default=3)
    p.add_argument('--output', default='results/memory_gate_soma.json')
    p.add_argument('--details', help='Optional complete per-trial receipt')
    args = p.parse_args()
    conditions = [benchmark(range(args.start_seed,args.start_seed+args.seeds),
                            MemorySetup(read_damage=eta), args.budget) for eta in (0.,.25)]
    if args.details:
        Path(args.details).parent.mkdir(parents=True, exist_ok=True)
        Path(args.details).write_text(json.dumps(conditions,indent=2)+'\n')
    for c in conditions:
        del c['runs']
        c['config']['seed_range'] = [c['config']['seeds'][0], c['config']['seeds'][-1]]
        del c['config']['seeds']
    receipt = {'date':'2026-10-09', 'protocol':'NEURON_PROTOCOL.md',
               'model':'abstract history traces and scalar readout; known finite candidate dictionary',
               'conditions':conditions}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    for c in conditions:
        print('eta =',c['config']['setup']['read_damage'])
        for m,s in c['summary'].items():
            print(f"  {m:10s} {s['correct']:3d}/{s['total']}  logloss {s['mean_logloss']:.4f}  forecast {s['forecast_nrmse']:.3f}  damage {s['mean_disturbance']:.3f}")
        print(' ',c['gates'])


if __name__ == '__main__':
    main()
