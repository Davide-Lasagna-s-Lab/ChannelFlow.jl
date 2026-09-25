"""Plot measured one-step velocity differences; no simulations are run."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results = Path(__file__).resolve().parents[1] / "results"
with (results / "cpp.csv").open() as stream:
    rows = sorted((row for row in csv.DictReader(stream)
                   if row["field"] == "0" and int(row["N"]) > 8),
                  key=lambda row: int(row["N"]))
N = [int(row["N"]) for row in rows]
errors = [float(row["max_abs_error"]) for row in rows]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
ax.plot(N, errors, "o-", color="#c62828", markersize=4)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_ylim(1e-16, 1e-13)
ax.set_xticks(N, labels=[str(n) for n in N])
ax.set_xlabel(r"Resolved $N_x=N_z=N$ ($N_y=N+1$)")
ax.set_ylabel(r"Maximum velocity difference $\epsilon_u$")
ax.minorticks_off()
ax.grid(which="major", alpha=0.25)
for extension in ("svg", "png"):
    path = results / f"cpp.{extension}"
    fig.savefig(path)
    if extension == "svg":
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
plt.close(fig)
