"""Copy named top-level config keys from a donor design file into a base design file."""

import json
import sys
from pathlib import Path

base, donor, out, name, *keys = sys.argv[1:]
designs = json.loads(Path(base).read_text())
donors = json.loads(Path(donor).read_text())
if len(designs) != 1 or len(donors) != 1:
    raise SystemExit("expected one design in each file")
design = designs[0]
design["name"] = name
for key in keys:
    if key not in donors[0]["config"]:
        raise SystemExit(f"donor lacks config key {key}")
    design["config"][key] = donors[0]["config"][key]
Path(out).parent.mkdir(parents=True, exist_ok=True)
Path(out).write_text(json.dumps(designs, indent=2) + "\n")
print(out, name, keys)
