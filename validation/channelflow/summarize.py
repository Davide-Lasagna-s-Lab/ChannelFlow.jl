"""Extract measured one-step parity errors from the C++ driver's stderr logs."""
from pathlib import Path
import csv
import re
import sys
root = Path(sys.argv[1])
with (root.parent / 'cpp.csv').open('w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(['N', 'field', 'max_abs_error', 'gauge_offset', 'max_reference'])
    for path in sorted(root.glob('N*.txt'), key=lambda p: int(p.stem[1:])):
        for line in path.read_text().splitlines():
            values = dict(re.findall(r'(\w+)=([^ ]+)', line))
            if 'field' in values:
                writer.writerow([int(path.stem[1:])] + [values[k] for k in
                    ('field', 'max_abs_error', 'gauge_offset', 'max_reference')])
