"""Draw a dependency-free SVG from the measured simulator and frozen receipt."""
from pathlib import Path
import numpy as np
from lens_luotain_target import Setup, catalog, choose_probe, transport, twins


def polyline(y, x0, y0, w, h, color, lo=0, hi=1):
    ys = np.asarray(y, dtype=float)
    points = ' '.join(f'{x0 + i*w/max(1,len(ys)-1):.2f},{y0 + (hi-v)/(hi-lo)*h:.2f}' for i,v in enumerate(ys))
    return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.1" stroke-linejoin="round"/>'


def render():
    setup = Setup()
    a0 = transport(setup)
    xs = twins(a0, 6100)
    matrices = np.stack([transport(setup,m,shift) for m,shift in catalog(setup)])
    predicted = np.einsum('pws,hs->phw', matrices, xs)
    first = choose_probe(np.ones(8)/8, predicted, np.arange(30))
    mask, shift = catalog(setup)[first]
    aa,bb = xs[0],xs[1]
    free_a,free_b = a0 @ aa, a0 @ bb
    masked_a,masked_b = matrices[first] @ aa, matrices[first] @ bb
    from json import loads
    summary=loads(Path('results/summary.json').read_text())
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="500" viewBox="0 0 960 500">',
         '<rect width="960" height="500" fill="#101b2b" rx="18"/>',
         '<style>text{font-family:system-ui,Arial,sans-serif;fill:#d6e1ef} .tiny{font-size:13px;fill:#a6b7c9}.head{font-size:20px;font-weight:bold}</style>',
         '<text x="35" y="38" class="head">One hidden difference, two different lenses</text>',
         '<text x="35" y="60" class="tiny">1-D shadow-model illustration: blue and orange curves are alternative scenes, not two photos</text>',
         '<text x="44" y="104" class="tiny">Hidden scene candidates</text>',
         '<text x="353" y="104" class="tiny">No mask: indistinguishable</text>',
         '<text x="667" y="104" class="tiny">Selected occluder: disagreement appears</text>']
    for x in [35,348,662]:
        svg.append(f'<rect x="{x}" y="115" width="260" height="179" rx="8" fill="#192a40" stroke="#3b526b"/>')
    svg.extend([polyline(aa,50,136,230,130,'#58a6ff',.05,.95),polyline(bb,50,136,230,130,'#ffad68',.05,.95),
                polyline(free_a,363,136,230,130,'#58a6ff',.05,.95),polyline(free_b,363,136,230,130,'#ffad68',.05,.95),
                polyline(masked_a,677,136,230,130,'#58a6ff',-.05,.85),polyline(masked_b,677,136,230,130,'#ffad68',-.05,.85)])
    for i,v in enumerate(mask):
        svg.append(f'<rect x="{680+i*20}" y="275" width="18" height="8" fill="{("#e7ca82" if v else "#415268")}"/>')
    svg.extend([f'<text x="677" y="310" class="tiny">Mask index {first}, shift {shift:+.2f} m</text>',
                '<text x="35" y="347" class="head">Held-out identification after two new wall measurements</text>'])
    for j,(label,key,color) in enumerate([('Active','active','#5acbb5'),('Random','random','#77aaf4'),('Repeat','repeat','#b09abf')]):
        y=374+j*37; acc=summary[key]['accuracy']
        svg.append(f'<text x="36" y="{y+14}" font-size="15">{label}</text>')
        svg.append(f'<rect x="144" y="{y}" width="690" height="22" rx="6" fill="#23364a"/>')
        svg.append(f'<rect x="144" y="{y}" width="{690*acc:.1f}" height="22" rx="6" fill="{color}"/>')
        svg.append(f'<text x="848" y="{y+17}" font-size="15" font-weight="bold">{acc*100:.1f}%</text>')
    svg.append('</svg>')
    Path('results/receipt.svg').write_text('\n'.join(svg)+'\n')
    print('wrote results/receipt.svg, bytes',Path('results/receipt.svg').stat().st_size)

if __name__=='__main__':
    render()
