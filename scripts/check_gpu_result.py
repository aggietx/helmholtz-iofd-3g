"""Reject an incomplete solve; process exit status alone is not convergence."""
import math
import sys
from pathlib import Path

lines = [s for s in Path(sys.argv[1]).read_text().splitlines()
         if s.startswith('brick_olfd3g ')]
if len(lines) != 1:
    raise SystemExit('Expected one solver summary')
row = dict(s.split('=', 1) for s in lines[0].split()[1:])
residual = float(row['relative_residual'])
if row.get('converged') != 'true' or not math.isfinite(residual) or not 0 <= residual <= float(sys.argv[2]):
    raise SystemExit('Solve did not reach the true relative residual tolerance')
print('Validated true relative residual:', residual)
