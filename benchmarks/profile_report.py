"""Summarize CPU samples and CUDA kernel activity without conflating the two."""
from pathlib import Path
import csv
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / 'results/final'
lines = ['# Where timestep execution spends time', '',
         'Profiles cover warmed complete CNRK2 steps, separately from throughput timings.',
         'CPU percentages classify sampled DNS call stacks into exclusive categories. '
         'They are sampling estimates, not exact phase timers; FFT worker activity '
         'is not a Julia call stack. Consult the raw call tree for context.', '',
         'GPU percentages are fractions of summed device activity duration. They exclude '
         'host launch overhead and idle gaps; overlapping activities must not be '
         'interpreted as additive wall time. Raw logs also contain CUDA host API activity.', '']
for path in sorted(root.glob('profile-*.csv')):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    gpu = 'seconds' in (rows[0] if rows else {})
    key = 'seconds' if gpu else 'samples'
    total = sum(float(r[key]) for r in rows)
    lines += [f'## {path.stem}', '', f'[Raw data]({path.name}) · [Full profiler output]({path.stem}.txt)', '']
    if not total:
        lines += ['No samples captured; increase CHANNEL_PROFILE_STEPS.', '']
        continue
    lines += ['| Kernel / category | Share | '+('Total device ms | Calls |' if gpu else 'Samples |'),
              '|---|---:|'+('---:|---:|' if gpu else '---:|')]
    for row in sorted(rows, key=lambda r: float(r[key]), reverse=True)[:15]:
        name = row['name' if gpu else 'category'].replace('|', r'\|')
        value = float(row[key])
        detail = f'{1000*value:.3f} | {row["calls"]}' if gpu else row[key]
        lines.append(f'| `{name}` | {100*value/total:.1f}% | {detail} |')
    if gpu:
        lines += ['', 'Top 15 kernels shown; percentages use all captured device activities.']
    lines.append('')
(root / 'profiling.md').write_text('\n'.join(lines), encoding='utf-8')
