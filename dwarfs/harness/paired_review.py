"""Paired review of a candidate round against the incumbent on the frozen 302-case cohort.

Both inputs are `<design>-scored-cases.json` files written by `evolution.py report`.
Cases are joined by case_id; the denominators must be identical. The review reports
planet gains and losses, protected identities, control activity and eligibility, and
evaluates the sequential development adoption criteria. It never reads sealed outcomes.
"""

import argparse
import collections
import json
from pathlib import Path

PLANET_KINDS = ("real_planet", "synthetic_planet")
CATALOGUE_CONTROL_KINDS = ("native_eb", "catalogue_fp")
DEPTH_EDGES_PPM = (0, 100, 200, 400, 800, float("inf"))
MAGNITUDE_EDGES = (0, 16.5, 17.0, 17.5, 18.0, float("inf"))
FROZEN_KEYS = ("kind", "input_sha256", "array_key", "receipt_sha256", "preparation_status")


def load(path):
    rows = json.load(open(path))
    cases = {row["case_id"]: row for row in rows}
    if len(cases) != len(rows):
        raise ValueError(f"duplicate case ids in {path}")
    return cases


def check_frozen(incumbent, candidate):
    if set(incumbent) != set(candidate):
        raise ValueError("case sets differ: not a paired comparison")
    for case_id, row in incumbent.items():
        for key in FROZEN_KEYS:
            if row.get(key) != candidate[case_id].get(key):
                raise ValueError(f"{case_id}: {key} differs between rounds")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def label(edges, value, unit):
    for low, high in zip(edges, edges[1:]):
        if low <= value < high:
            top = "inf" if high == float("inf") else f"{high:g}"
            return f"{low:g}-{top} {unit}"
    return "unknown"


def planet_rows(incumbent, candidate):
    rows = []
    for case_id, row in incumbent.items():
        if row["kind"] not in PLANET_KINDS:
            continue
        other = candidate[case_id]
        rows.append(
            {
                "case_id": case_id,
                "kind": row["kind"],
                "population": row.get("population", ""),
                "depth_ppm": number(row.get("depth_ppm")),
                "phot_g_mean_mag": number(row.get("phot_g_mean_mag")),
                "incumbent": bool(row["recovered"]),
                "candidate": bool(other["recovered"]),
                "incumbent_proposals": row["proposal_count"],
                "candidate_proposals": other["proposal_count"],
                "incumbent_status": row["detector_status"],
                "candidate_status": other["detector_status"],
                "incumbent_first_failure": row.get("detector_first_failure"),
                "candidate_first_failure": other.get("detector_first_failure"),
            }
        )
    return rows


def strata(rows):
    keys = {
        "kind": lambda r: r["kind"],
        "population": lambda r: r["population"],
        "depth": lambda r: label(DEPTH_EDGES_PPM, r["depth_ppm"], "ppm"),
        "magnitude": lambda r: label(MAGNITUDE_EDGES, r["phot_g_mean_mag"], "G"),
    }
    tables = {}
    for name, key in keys.items():
        table = collections.defaultdict(
            lambda: {"count": 0, "incumbent": 0, "candidate": 0, "gains": 0, "losses": 0}
        )
        for r in rows:
            cell = table[key(r)]
            cell["count"] += 1
            cell["incumbent"] += r["incumbent"]
            cell["candidate"] += r["candidate"]
            cell["gains"] += r["candidate"] and not r["incumbent"]
            cell["losses"] += r["incumbent"] and not r["candidate"]
        tables[name] = dict(sorted(table.items()))
    return tables


def planet_review(incumbent, candidate):
    rows = planet_rows(incumbent, candidate)
    gains = [r for r in rows if r["candidate"] and not r["incumbent"]]
    losses = [r for r in rows if r["incumbent"] and not r["candidate"]]
    return {
        "planets": len(rows),
        "incumbent_recovered": sum(r["incumbent"] for r in rows),
        "candidate_recovered": sum(r["candidate"] for r in rows),
        "net_gain": len(gains) - len(losses),
        "gains": gains,
        "losses": losses,
        "protected_identities": sum(r["incumbent"] for r in rows),
        "protected_lost": [r["case_id"] for r in losses],
        "protected_strata": strata([r for r in rows if r["incumbent"]]),
        "strata": strata(rows),
    }


def control_review(incumbent, candidate):
    per_kind = {}
    for case_id, row in incumbent.items():
        if row["kind"] in PLANET_KINDS:
            continue
        other = candidate[case_id]
        cell = per_kind.setdefault(
            row["kind"],
            {
                "count": 0,
                "incumbent_active": 0,
                "candidate_active": 0,
                "incumbent_proposals": 0,
                "candidate_proposals": 0,
                "changed_cases": [],
            },
        )
        cell["count"] += 1
        cell["incumbent_active"] += row["proposal_count"] > 0
        cell["candidate_active"] += other["proposal_count"] > 0
        cell["incumbent_proposals"] += row["proposal_count"]
        cell["candidate_proposals"] += other["proposal_count"]
        changed = (
            row["proposal_count"] != other["proposal_count"]
            or row["detector_status"] != other["detector_status"]
        )
        if changed:
            cell["changed_cases"].append(
                {
                    "case_id": case_id,
                    "incumbent_proposals": row["proposal_count"],
                    "candidate_proposals": other["proposal_count"],
                    "incumbent_status": row["detector_status"],
                    "candidate_status": other["detector_status"],
                }
            )
    counters = (
        "count",
        "incumbent_active",
        "candidate_active",
        "incumbent_proposals",
        "candidate_proposals",
    )
    totals = {key: sum(cell[key] for cell in per_kind.values()) for key in counters}
    catalogue = {
        key: sum(per_kind[kind][key] for kind in CATALOGUE_CONTROL_KINDS if kind in per_kind)
        for key in ("incumbent_active", "candidate_active")
    }
    return {"per_kind": per_kind, "totals": totals, "catalogue_any_proposal": catalogue}


def eligibility_review(incumbent, candidate):
    changes = []
    for case_id, row in incumbent.items():
        other = candidate[case_id]
        if row["detector_status"] != other["detector_status"]:
            changes.append(
                {
                    "case_id": case_id,
                    "kind": row["kind"],
                    "incumbent_status": row["detector_status"],
                    "candidate_status": other["detector_status"],
                    "candidate_first_failure": other.get("detector_first_failure"),
                }
            )
    regressions = [
        c for c in changes if c["incumbent_status"] == "SEARCHED" and c["candidate_status"] != "SEARCHED"
    ]
    return {"status_changes": changes, "searched_regressions": regressions}


def criteria(planets, controls, eligibility):
    totals, catalogue = controls["totals"], controls["catalogue_any_proposal"]
    return {
        "planet_net_gain_at_least_one": planets["net_gain"] >= 1,
        "no_protected_identity_lost": not planets["protected_lost"],
        "no_searched_eligibility_regression": not eligibility["searched_regressions"],
        "no_catalogue_control_any_proposal_increase": catalogue["candidate_active"]
        <= catalogue["incumbent_active"],
        "no_control_any_activity_increase": totals["candidate_active"] <= totals["incumbent_active"],
        "no_control_total_proposal_increase": totals["candidate_proposals"]
        <= totals["incumbent_proposals"],
    }


def describe(r):
    return (
        f"{r['case_id']} {r['kind']} {r['population']} {r['depth_ppm']:.0f} ppm "
        f"G{r['phot_g_mean_mag']:.2f} proposals {r['incumbent_proposals']}->{r['candidate_proposals']} "
        f"status {r['incumbent_status']}->{r['candidate_status']}"
    )


def print_summary(review):
    planets, controls, eligibility = review["planets"], review["controls"], review["eligibility"]
    print(f"planets {planets['planets']}: recovered {planets['incumbent_recovered']} -> "
          f"{planets['candidate_recovered']} (net {planets['net_gain']:+d})")
    for name in ("gains", "losses"):
        print(f"  {name} {len(planets[name])}")
        for r in planets[name]:
            print("    " + describe(r))
    print(f"  protected identities {planets['protected_identities']}, lost {planets['protected_lost']}")
    for name, table in planets["strata"].items():
        print(f"  by {name}:")
        for key, cell in table.items():
            print(f"    {key:>14} n={cell['count']:3d} {cell['incumbent']:2d} -> {cell['candidate']:2d}"
                  f" (+{cell['gains']} -{cell['losses']})")
    totals = controls["totals"]
    print(f"controls {totals['count']}: any-activity {totals['incumbent_active']} -> "
          f"{totals['candidate_active']}, proposals {totals['incumbent_proposals']} -> "
          f"{totals['candidate_proposals']}")
    cat = controls["catalogue_any_proposal"]
    print(f"  native_eb+catalogue_fp any-proposal {cat['incumbent_active']} -> {cat['candidate_active']}")
    for kind, cell in sorted(controls["per_kind"].items()):
        print(f"    {kind:>38} n={cell['count']:3d} active {cell['incumbent_active']:2d} -> "
              f"{cell['candidate_active']:2d} proposals {cell['incumbent_proposals']:3d} -> "
              f"{cell['candidate_proposals']:3d} changed {len(cell['changed_cases'])}")
    print(f"eligibility: status changes {len(eligibility['status_changes'])}, "
          f"searched regressions {len(eligibility['searched_regressions'])}")
    for c in eligibility["status_changes"]:
        print(f"    {c['case_id']} {c['kind']} {c['incumbent_status']} -> {c['candidate_status']}")
    print("criteria:")
    for name, passed in review["criteria"].items():
        print(f"    {'PASS' if passed else 'FAIL'} {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    incumbent, candidate = load(args.incumbent), load(args.candidate)
    check_frozen(incumbent, candidate)
    planets = planet_review(incumbent, candidate)
    controls = control_review(incumbent, candidate)
    eligibility = eligibility_review(incumbent, candidate)
    review = {
        "incumbent": str(args.incumbent),
        "candidate": str(args.candidate),
        "cases": len(incumbent),
        "planets": planets,
        "controls": controls,
        "eligibility": eligibility,
        "criteria": criteria(planets, controls, eligibility),
    }
    args.out.write_text(json.dumps(review, indent=2, sort_keys=True, allow_nan=True))
    print_summary(review)


if __name__ == "__main__":
    main()
