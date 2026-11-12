"""Write a one-design file from a base design with dotted config changes (key.path=json)."""

import json
import sys
from pathlib import Path

base, out, name, *changes = sys.argv[1:]
designs = json.loads(Path(base).read_text())
if len(designs) != 1:
    raise SystemExit("expected one base design")
design = designs[0]
design["name"] = name
for change in changes:
    path, value = change.split("=", 1)
    node = design["config"]
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    if keys[-1] not in node:
        raise SystemExit(f"unknown config key {path}")
    node[keys[-1]] = json.loads(value)
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(designs, indent=2) + "\n")
print(out, name, changes)
