"""Join the three stages' evidence; a vetting dismissal stands, nothing is accepted."""

from common import read, receipt, require, write


def collect(destination):
    vetted = read(destination / "vetting/vetting.json")
    pixel_path = destination / "localization/localization.json"
    pixels = read(pixel_path) if pixel_path.exists() else None
    by_id = {row["signal_id"]: row for row in pixels["results"]} if pixels else {}
    if pixels:
        require(
            pixels["source_id"] == vetted["source_id"], "BOI source identity differs"
        )
        require(
            pixels["vetting"]["sha256"]
            == receipt(destination / "vetting/vetting.json")["sha256"],
            "BOI localization used different vetting",
        )
        require(
            set(by_id) == {row["signal_id"] for row in vetted["results"]},
            "BOI signals differ",
        )
    results = []
    for index, row in enumerate(vetted["results"], 1):
        localization = by_id.get(row["signal_id"], {"state": "NOT_RUN"})
        state = row["disposition"]["state"]
        # Pixels cannot rescue a dismissed signal; others stay unresolved until localized.
        if state != "DISMISSED" and localization["state"] != "MEASURED":
            state = "UNRESOLVED"
        results.append(
            {
                "boi_id": f"BOI-{vetted['source_id']}-{index}",
                "signal_id": row["signal_id"],
                "state": state,
                "vetting": row,
                "localization": localization,
            }
        )
    path = destination / "bois.json"
    write(
        path,
        {
            "source_id": vetted["source_id"],
            "decision_policy_applied": bool(vetted.get("decision_policy_applied")),
            "scope": "Provisional BOI evidence; no calibrated acceptance decision or statistical validation",
            "vetting": receipt(destination / "vetting/vetting.json"),
            "localization": receipt(pixel_path) if pixels else None,
            "results": results,
        },
    )
    return receipt(path)
