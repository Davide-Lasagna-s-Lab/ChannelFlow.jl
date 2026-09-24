"""Plot serial C++ versus serial Julia on matching resolved grids and hardware."""
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cpp_path = Path(sys.argv[1])
julia_path = Path(sys.argv[2])
def read(path):
    with path.open() as f:
        return {int(r['Nx']): float(r['min_seconds']) for r in csv.DictReader(f)}
cpp, julia = read(cpp_path), read(julia_path)
sizes = sorted(cpp.keys() & julia.keys())
fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
axes[0].loglog(sizes, [1000*cpp[n] for n in sizes], 'o-', label='C++ serial')
axes[0].loglog(sizes, [1000*julia[n] for n in sizes], 's-', label='Julia serial')
axes[0].set_ylabel('Complete CNRK2 step [ms]')
axes[0].legend(frameon=False)
axes[1].loglog(sizes, [cpp[n]/julia[n] for n in sizes], 'o-')
axes[1].axhline(1, color='0.5', lw=0.8)
axes[1].set_ylabel('C++ time / Julia time')
for ax in axes:
    ax.set_xlabel(r'Resolved $N_x=N_z$ ($N_y=N_x+1$)')
    ax.set_xticks(sizes, labels=sizes)
    ax.grid(alpha=0.2)
    ax.spines[['top', 'right']].set_visible(False)
for ext in ('svg', 'png'):
    fig.savefig(cpp_path.parent / f'cpp-julia.{ext}', dpi=200)
