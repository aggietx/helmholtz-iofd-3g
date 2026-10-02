"""Check source hashes, local includes, and accidental private material."""
import argparse
import hashlib
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--write-manifest', action='store_true')
args = parser.parse_args()
sources = sorted(p for p in (root / 'src').rglob('*')
                 if p.suffix in ('.cpp', '.cu', '.hpp', '.inc'))
manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources}
target = root / 'docs' / 'source-sha256.json'
if args.write_manifest:
    target.write_text(json.dumps(manifest, indent=2) + '\n')
else:
    if manifest != json.loads(target.read_text()):
        raise SystemExit('Source manifest mismatch')
for path in sources:
    for name in re.findall(r'^\s*#include\s+"([^"]+)"', path.read_text(), re.M):
        if not (path.parent / name).is_file():
            raise SystemExit(f'Missing local include: {path.name}: {name}')
for path in root.rglob('*'):
    if not path.is_file() or any(p in ('.git', 'build', 'results', '__pycache__')
                                 for p in path.relative_to(root).parts):
        continue
    if path == Path(__file__).resolve():
        continue
    data = path.read_text(errors='replace')
    for pattern in (r'-----BEGIN .*PRIVATE KEY-----', r'gh[pousr]_[A-Za-z0-9]{20,}',
                    r'/Users/', r'/HOME/', r'/home/bingxing',
                    r'ssh\.cn-zhongwei', r'pxy785', r'scx7758'):
        if re.search(pattern, data):
            raise SystemExit(f'Private material in {path.relative_to(root)}')
print(f'PASS: {len(sources)} source files; hashes, includes, and private-material scan')
