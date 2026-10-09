"""Generate browser examples and the exact held-out receipt, with no network."""
from dataclasses import asdict
import json
from pathlib import Path

from memory_gate_soma import (MemorySetup, scene, decay_constants, gate_catalog,
                              uniform_gate, forecast_gates)


def main():
    root=Path(__file__).resolve().parent
    setup=MemorySetup()
    config={**asdict(setup),'budget':3,'decays':decay_constants(setup).tolist(),
            'gates':gate_catalog(setup).tolist(),'uniform':uniform_gate(setup).tolist(),
            'forecast_gates':forecast_gates(setup).tolist()}
    examples=[]
    for seed in (8100,8101,8102):
        s=scene(seed,setup)
        examples.append({'seed':seed,'histories':s['histories'].tolist(),
                         'truth':s['truth'],'initial_noise':s['initial_noise'],
                         'noise':s['noise'][:3].tolist(),'random_order':s['random_order'].tolist()})
    payload={'config':config,'scenes':examples,
             'receipt':json.loads((root/'results/memory_gate_soma.json').read_text())}
    source="'use strict';\nconst MemoryGateData="+json.dumps(payload,separators=(',',':'))+";\n"
    source+="if(typeof module!=='undefined'&&module.exports)module.exports=MemoryGateData;else globalThis.MemoryGateData=MemoryGateData;\n"
    (root/'site/neuron-data.js').write_text(source)


if __name__=='__main__':
    main()
