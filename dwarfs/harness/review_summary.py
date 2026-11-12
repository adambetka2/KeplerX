"""Print the headline of a paired review file."""

import json
import sys

d = json.load(open(sys.argv[1]))
p, c = d["planets"], d["controls"]
print("incumbent", d["incumbent"].split("/")[-2][:70])
print("candidate", d["candidate"].split("/")[-2][:70])
print("recovered", p["incumbent_recovered"], "->", p["candidate_recovered"], "net", p["net_gain"])
print("gains", json.dumps(p["gains"])[:600])
print("losses", json.dumps(p["losses"])[:600])
print("protected_lost", json.dumps(p["protected_lost"])[:300])
print("strata", json.dumps(p["strata"])[:500])
print("controls totals", json.dumps(c["totals"])[:400])
print("catalogue_any_proposal", json.dumps(c["catalogue_any_proposal"])[:300])
print("criteria", json.dumps(d["criteria"]))
print("eligibility", json.dumps(d["eligibility"])[:300])
