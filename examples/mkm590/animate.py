"""Animate total streamwise velocity at x=0 from saved (z,y) slices."""
from pathlib import Path
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if os.environ.get("FFMPEG"):
    plt.rcParams["animation.ffmpeg_path"] = os.environ["FFMPEG"]
from matplotlib.animation import FuncAnimation, FFMpegWriter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--fps", type=int, default=20)
args = parser.parse_args()
files = sorted(args.output.glob("slice-*.bin"))[::args.stride]
if not files:
    raise ValueError("No saved slices found")

def read(path):
    with path.open("rb") as f:
        nz, ny = np.fromfile(f,dtype="<i8",count=2)
        t = np.fromfile(f,dtype="<f8",count=1)[0]
        z = np.fromfile(f,dtype="<f8",count=nz)
        y = np.fromfile(f,dtype="<f8",count=ny)
        u = np.fromfile(f,dtype="<f8",count=nz*ny).reshape((nz,ny),order="F")
    # Close the periodic boundary; pcolormesh respects the nonuniform y nodes.
    return t, np.r_[z,z[-1]+z[1]-z[0]], y, np.vstack((u,u[0]))

t,z,y,u = read(files[0])
fig,ax = plt.subplots(figsize=(8,5),layout="constrained")
mesh=ax.pcolormesh(z,y,u.T,shading="gouraud",cmap="turbo",vmin=0,vmax=1.6,rasterized=True)
ax.set(xlabel=r"$z/h$",ylabel=r"$y/h$",aspect="equal",ylim=(-1,1),xlim=(0,z[-1]))
cb=fig.colorbar(mesh,ax=ax,pad=.025)
cb.set_label(r"$u/U_b$")
time_label=ax.set_title("")
def update(i):
    t,z,y,u=read(files[i])
    mesh.set_array(u.T.ravel())
    time_label.set_text(rf"$x=0,\quad tU_b/h={t:.2f}$")
    return mesh,time_label
update(len(files)-1)
fig.savefig(args.output/"snapshot.png",dpi=300)
fig.savefig(args.output/"snapshot.pdf")
movie=FuncAnimation(fig,update,frames=len(files),interval=1000/args.fps,blit=False)
movie.save(args.output/"streamwise-yz.mp4",writer=FFMpegWriter(fps=args.fps),dpi=150)
print(f"Saved {len(files)} frames and the final snapshot to {args.output}")
