"""Compare saved plane/time statistics with MKM590; never start a DNS here."""
from pathlib import Path
import argparse
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebder, chebval

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="DNS output directory containing profiles.csv")
args = parser.parse_args()
reference = Path(__file__).parent / "reference"
mean = np.loadtxt(reference / "chan590.means", comments="#")
stress = np.loadtxt(reference / "chan590.reystress", comments="#")
a = np.genfromtxt(args.output / "profiles.csv", delimiter=",", names=True)
y, U = a["y"], a["U"]
# Fit on exactly the simulation's Lobatto nodes; evaluate mean wall shear.
c = chebfit(y, U, len(y)-1)
d = chebder(c)
config = tomllib.loads((args.output / "config.toml").read_text())
nu = config["nu"]
utau = np.sqrt(nu*(chebval(-1,d)-chebval(1,d))/2)
if not np.isfinite(utau) or utau <= 0:
    raise ValueError("Mean wall stress is not positive and finite")
# Fold the two walls. A wall-normal component reverses sign under reflection;
# therefore uv and vw are antisymmetric, whereas normal stresses are symmetric.
idx = np.flatnonzero(y <= 1e-14)
mirror = len(y)-1-idx
yp = (1+y[idx])*utau/nu
order = np.argsort(yp)
yp = yp[order]
fold = lambda name, parity=1: (0.5*(a[name][idx]+parity*a[name][mirror]))[order]
plt.rcParams.update({"figure.figsize":(10,4), "figure.dpi":150,
                     "savefig.dpi":300, "axes.spines.top":False,
                     "axes.spines.right":False, "xtick.direction":"out", "ytick.direction":"out"})
fig, axes = plt.subplots(1,2, layout="constrained")
axes[0].semilogx(yp[yp>0], (fold("U")/utau)[yp>0], color="#4063d8", label="ChannelFlow.jl")
axes[0].semilogx(mean[1::6,1], mean[1::6,2], "o", ms=3, mfc="none", color="black", label="Moser–Kim–Mansour MKM590")
axes[0].set(xlabel=r"$y^+$", ylabel=r"$U^+$")
axes[0].legend(frameon=False)
colors = ("#4063d8", "#389826", "#9558b2", "#cb3c33")
for k,(name,label) in enumerate(zip(("uu","vv","ww","uv"),
        (r"$\overline{u'u'}^+$",r"$\overline{v'v'}^+$",r"$\overline{w'w'}^+$",r"$-\overline{u'v'}^+$"))):
    sign = -1 if name == "uv" else 1
    values = sign*fold(name,sign)/utau**2
    axes[1].plot(yp,values,color=colors[k],label=label)
    axes[1].plot(stress[::6,1],sign*stress[::6,k+2],"o",ms=3,mfc="none",color=colors[k])
axes[1].set(xlabel=r"$y^+$",ylabel="Reynolds stresses")
axes[1].legend(frameon=False,ncol=2)
for ax in axes:
    ax.grid(alpha=.2)
fig.savefig(args.output / "comparison.png")
fig.savefig(args.output / "comparison.pdf")
print(f"Mean-wall-stress Re_tau = {utau/nu:.6g}; reference = 587.19")
print("Lines: current DNS statistics; open circles: Moser–Kim–Mansour. Check averaging convergence before interpreting differences.")
