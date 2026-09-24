"""Render validation figures from measured CSVs; no simulations run here."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).parent / 'results'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 200})
def read(name):
    with (root / f'{name}.csv').open() as f:
        return list(csv.DictReader(f))
def val(rows, key):
    return np.array([float(r[key]) for r in rows])
def save(fig, name):
    for ax in fig.axes:
        ax.grid(alpha=.2)
    for ext in ('svg', 'png'):
        path = root / f'{name}.{ext}'
        fig.savefig(path)
        if ext == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)

rows = read('decay')
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
for n in (1,3):
    subset = [r for r in rows if r['flow']=='Couette' and int(r['n'])==n and float(r['dt'])==.125]
    t = np.linspace(0,4,100)
    ax[0].plot(t, np.exp(-2*float(subset[0]['mu'])*t), label=f'Exact n={n}')
    ax[0].plot(val(subset,'t'), val(subset,'energy_ratio'), 'o', mfc='none')
for flow, marker in [('Couette','o'),('Poiseuille','x')]:
    for n in (1,3):
        subset = sorted([r for r in rows if r['flow']==flow and int(r['n'])==n and float(r['t'])==4], key=lambda r: float(r['dt']))
        ax[1].loglog(val(subset,'dt'),val(subset,'error'),marker+'-',label=f'{flow}, n={n}')
dt=np.array([.125,.5]); ax[1].loglog(dt, 2e-5*dt**2,'k--',label=r'$C\Delta t^2$')
ax[0].set(xlabel=r'$t$',ylabel=r"$E'(t)/E'(0)$")
ax[1].set(xlabel=r'$\Delta t$',ylabel='Final relative field error')
for a in ax: a.legend(fontsize=8,frameon=False)
save(fig,'decay')

rows=read('ts_history'); conv=read('ts_convergence')
fig, ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
t=np.linspace(0,50,100); ax[0].plot(t,np.exp(2*.002664410371*t),'k-',label='Linear reference')
for dt in (.4,.2,.1):
    subset=[r for r in rows if float(r['dt'])==dt]
    ax[0].plot(val(subset,'t'),val(subset,'energy_ratio'),'o',mfc='none',label=rf'$\Delta t={dt}$')
ax[1].loglog(val(conv,'dt'),val(conv,'error'),'o-',label='DNS')
dt=np.array([.1,.4]); ax[1].loglog(dt,float(conv[-1]['error'])*(dt/.1)**2,'k--',label=r'$C\Delta t^2$')
ax[0].set(xlabel=r'$t$',ylabel=r"$E'(t)/E'(0)$")
ax[1].set(xlabel=r'$\Delta t$',ylabel='Final relative coefficient error')
for a in ax: a.legend(fontsize=8,frameon=False)
save(fig,'ts')

rows=read('waleffe')
fig, ax=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
for branch,color in [('LB','#1565c0'),('UB','#c62828')]:
    for n,marker in [(32,'o'),(48,'s')]:
        subset=[r for r in rows if r['branch']==branch and int(r['N'])==n and float(r['dt'])==.0125]
        label=f'{branch}, N={n}'
        ax[0].semilogy(val(subset,'t'),val(subset,'drift'),marker+'-',color=color,label=label)
        imbalance=np.abs(val(subset,'input')/val(subset,'dissipation')-1)
        ax[1].semilogy(val(subset,'t'),imbalance,marker+'-',color=color,label=label)
ax[0].set(xlabel=r'$t$',ylabel=r'$\|u(t)-u(0)\|/\|u(0)\|$')
ax[1].set(xlabel=r'$t$',ylabel=r'$|I/D-1|$')
for a in ax: a.legend(fontsize=8,frameon=False)
save(fig,'waleffe')

rows=read('cpp')
fig, ax=plt.subplots(figsize=(5,3.6),layout='constrained')
for field,label in [('0','Velocity'),('1','Modified pressure (gauge removed)')]:
    subset=[r for r in rows if r['field']==field]
    ax.semilogy(val(subset,'N'),val(subset,'max_abs_error'),'o-',label=label)
ax.set(xlabel=r'Resolved $N_x=N_z$ ($N_y=N_x+1$)',ylabel='Maximum absolute one-step difference',xticks=[8,16,32])
ax.legend(fontsize=8,frameon=False)
save(fig,'cpp')
