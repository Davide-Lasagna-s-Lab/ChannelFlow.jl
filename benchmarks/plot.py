"""Render committed CSV timings and sampled CPU stacks (requires matplotlib)."""
from pathlib import Path
import csv
import html
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / "results"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})

def rows(name):
    with (root / name).open() as f:
        return list(csv.DictReader(f))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
series = [("xeon-1.csv", "Host CPU, 1 thread", "#1565c0", "o"),
          ("xeon-4.csv", "Host CPU, 4 FFT threads", "#00897b", "s"),
          ("a100.csv", "A100 GPU", "#c62828", "^")]
data = {}
for name, label, color, marker in series:
    if not (root / name).exists():
        continue
    data[name] = {int(r["Ny"]): float(r["min_seconds"]) for r in rows(name)}
    x = sorted(data[name])
    axes[0].loglog(x, [1000*data[name][n] for n in x], color=color,
                   marker=marker, markersize=4, label=label)
if "a100.csv" in data:
    for name, label, color, marker in series[:2]:
        if name not in data:
            continue
        x = sorted(data[name].keys() & data["a100.csv"].keys())
        axes[1].semilogx(x, [data[name][n]/data["a100.csv"][n] for n in x],
                         color=color, marker=marker, markersize=4, label=label)
axes[0].set_ylabel("Complete timestep [ms]")
axes[1].set_ylabel("CPU time / GPU time")
axes[1].axhline(1, color="0.5", lw=.8)
for ax in axes:
    ax.set_xlabel(r"$N_y$  ($N_x=N_z=N_y-1$)")
    ticks = [9,17,33,65,129,257] if ax is axes[0] else [9,17,33,65,129]
    ax.set_xticks(ticks, labels=ticks)
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
fig.savefig(root / "timestep-cost.svg")
fig.savefig(root / "timestep-cost.png")
plt.close(fig)

# Keep the matched before/after experiment separate from the older CPU/GPU
# sweep: combining revisions would imply a comparison we did not measure.
if (root / "a100-broadcast-after.csv").exists():
    before = {int(r["Ny"]): float(r["min_seconds"])
              for r in rows("a100-broadcast-before.csv")}
    after = {int(r["Ny"]): float(r["min_seconds"])
             for r in rows("a100-broadcast-after.csv")}
    x = sorted(before.keys() & after.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for values, label, marker, color in (
        (before, "Before fusion", "o", "#757575"),
        (after, "Fused transforms", "^", "#c62828"),
    ):
        axes[0].loglog(x, [1000*values[n] for n in x],
                       marker=marker, color=color, markersize=4, label=label)
    axes[1].semilogx(x, [before[n]/after[n] for n in x],
                    marker="^", color="#c62828", markersize=4)
    axes[1].axhline(1, color="0.5", lw=.8)
    axes[0].set_ylabel("Complete timestep [ms]")
    axes[1].set_ylabel("Before / after time")
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel(r"$N_y$  ($N_x=N_z=N_y-1$)")
        ax.set_xticks(x, labels=x)
        ax.minorticks_off()
        ax.grid(alpha=.2)
    fig.savefig(root / "broadcast-fusion.svg")
    fig.savefig(root / "broadcast-fusion.png")
    plt.close(fig)

for name in ("apple-m5-phases", "xeon-phases"):
    path = root / (name + ".csv")
    if not path.exists():
        continue
    values = rows(path.name)
    total = sum(int(r["samples"]) for r in values)
    fig, ax = plt.subplots(figsize=(8,4), layout="constrained")
    ax.barh([r["category"] for r in values],
            [100*int(r["samples"])/total for r in values], color="#7b1fa2")
    ax.invert_yaxis()
    ax.set_xlabel("Share of CPU samples [%] (each sample counted once)")
    fig.savefig(root / (name + ".svg"))
    plt.close(fig)

    # A standalone flame graph retains the full call stack on hover.
    tree = {"count": 0, "children": {}}
    for line in path.with_suffix(".folded").read_text().splitlines():
        stack, count = line.rsplit(" ", 1)
        count = int(count)
        labels = stack.split(";")
        start = next((i for i,s in enumerate(labels) if s.startswith("step!")), 0)
        node = tree
        node["count"] += count
        for label in labels[start:]:
            node = node["children"].setdefault(label, {"count":0,"children":{}})
            node["count"] += count
    rectangles = []
    depth_max = [0]
    def draw(node, x, width, depth):
        for label, child in node["children"].items():
            w = width * child["count"] / node["count"]
            if w >= 2:
                depth_max[0] = max(depth_max[0], depth)
                hue = int(hashlib.md5(label.encode()).hexdigest()[:4],16) % 50
                title = html.escape(f"{label}: {child['count']} samples")
                text = html.escape(label[:max(0,int(w/7)-2)])
                rectangles.append(f'<g><title>{title}</title><rect x="{x:.2f}" y="{35+18*depth}" width="{w:.2f}" height="17" fill="hsl({hue},75%,65%)"/><text x="{x+3:.2f}" y="{48+18*depth}" font-size="11">{text}</text></g>')
                draw(child,x,w,depth+1)
            x += w
    draw(tree,0,1200,0)
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{65+18*depth_max[0]}" font-family="sans-serif"><rect width="100%" height="100%" fill="white"/><text x="8" y="20">CPU sampled call stacks — hover for function and sample count</text>'+''.join(rectangles)+'</svg>'
    (root / (name + "-flame.svg")).write_text(svg)


if (root / "gpu-kernels.csv").exists():
    groups = {}
    for r in rows("gpu-kernels.csv"):
        name = r["name"]
        if "_solve_kernel_" in name:
            category = "Batched Helmholtz solves"
        elif "fft" in name:
            category = "Fourier and Chebyshev FFTs"
        elif any(k in name for k in ("_derivative_", "_fourier_", "_curl_")):
            category = "Spectral derivatives"
        elif "_influence_" in name or "_tau_" in name:
            category = "Influence and tau"
        elif "_mean_" in name:
            category = "Mean-mode constraints"
        elif "copy" in name or "fill" in name:
            category = "Device copies and fills"
        else:
            category = "Broadcast kernels"
        groups[category] = groups.get(category,0) + float(r["seconds"])
    groups = dict(sorted(groups.items(),key=lambda p:p[1],reverse=True))
    with (root / "gpu-phases.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(["category","seconds_three_steps"])
        writer.writerows(groups.items())
    fig, ax = plt.subplots(figsize=(8,4), layout="constrained")
    ax.barh(list(groups),[1000*v/3 for v in groups.values()],color="#c62828")
    ax.invert_yaxis()
    ax.set_xlabel("Summed device time per timestep [ms] (three profiled steps)")
    fig.savefig(root / "gpu-kernels.svg")
    fig.savefig(root / "gpu-kernels.png")
    plt.close(fig)
