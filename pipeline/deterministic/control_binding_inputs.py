"""Bind retained opened preparations without replaying their measurements."""

from pathlib import Path


def preparation_sources(manifest, preparation, opened):
    from pipeline.deterministic.control_bindings import require

    require(
        manifest.get("scope") == "OPENED_DEVELOPMENT_ONLY"
        and manifest.get("role") == "development"
        and manifest.get("original_opened876_selection_unchanged") is True,
        "Retained preparation requires the unchanged opened development scope",
    )
    rows = preparation["sources"]
    ids = [row["source_id"] for row in rows]
    require(
        len(ids) == len(set(ids))
        and set(ids) == set(manifest["sources"])
        and set(ids) <= opened.keys(),
        "Retained preparation source identities differ",
    )
    return rows


def source_binding(source, supplied, declared):
    from pipeline.deterministic.control_bindings import require

    expected = {
        "source_id": declared["source_id"],
        "raw_fits_sha256": declared["raw_fits_sha256"],
        "origin_tpf_kic": int(declared["parent_tpf"]),
        "shard": declared["shard"],
    }
    require(
        all(supplied["metadata"].get(key) == value for key, value in expected.items())
        and source["parent_tpf"] == declared["parent_tpf"]
        and source["raw_fits_sha256"] == declared["raw_fits_sha256"]
        and source["shard"] == declared["shard"]
        and supplied["opened_cohort"] == declared["cohort"]
        and supplied["origin"]["original_array"]["sha256"]
        == declared["cloud_arrays_sha256"],
        "Retained source/parent/raw/original-array binding changed",
    )


def implementation_differences(frozen, root):
    from pipeline.deterministic.control_bindings import digest, require

    current, differences = {}, {}
    for relative, expected in frozen.items():
        relative_path = Path(relative)
        require(
            not relative_path.is_absolute()
            and ".." not in relative_path.parts
            and relative_path.parts[0] == "pipeline",
            "Measured code path escapes pipeline",
        )
        path = root / relative
        actual = digest(path) if path.is_file() else None
        current[relative] = actual
        if actual != expected:
            differences[relative] = {"measured": expected, "current": actual}
    return current, differences


def prepared_inputs(source, manifest_path, supplied):
    from pipeline.deterministic.control_bindings import (
        reference,
        require,
        verified_reference,
    )

    require(
        source.get("status") == "PREPARED"
        and set(source["inputs"]) | set(source["unavailable_views"])
        == {"psf_flat", "psf_qdiag", "sap_qdiag", "psf_nova"},
        "Retained prepared view inventory is incomplete",
    )
    relative = Path(supplied["arrays"])
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Retained opened array path escapes its manifest",
    )
    path = verified_reference(
        {"path": str(manifest_path.parent / relative), "sha256": supplied["sha256"]}
    )
    identities = [reference(path)]
    views = {}
    for view, row in source["inputs"].items():
        require(row["view"] == view, "Prepared view identity changed")
        path = verified_reference(
            {"path": row["fsc_path"], "sha256": row["fsc_sha256"]}
        )
        views[view] = {**row, "file": reference(path)}
        identities.append(reference(path))
    return views, identities


def bind_preparations(opened_rows, references, root):
    from pipeline.deterministic.control_bindings import (
        read_json,
        reference,
        require,
        verified_reference,
    )

    require(
        isinstance(references, list), "Opened preparation references must be a list"
    )
    opened = {row["source_id"]: row for row in opened_rows}
    sources, runs, identities = {}, [], []
    for ref in references:
        require(
            set(ref) == {"manifest", "preparation"},
            "Opened preparation reference fields changed",
        )
        manifest_path, preparation_path = (
            verified_reference(ref[key]) for key in ("manifest", "preparation")
        )
        manifest, preparation = read_json(manifest_path), read_json(preparation_path)
        require(
            preparation["input_manifest"]["sha256"] == ref["manifest"]["sha256"],
            "Preparation and original opened manifest identity differ",
        )
        identities.extend([reference(manifest_path), reference(preparation_path)])
        rows = preparation_sources(manifest, preparation, opened)
        current, differences = implementation_differences(
            preparation["implementation"], root
        )
        for source in rows:
            sid = source["source_id"]
            require(sid not in sources, "Retained preparation repeats an opened source")
            source_binding(source, manifest["sources"][sid], opened[sid])
            require(
                not opened[sid]["declared_raw_exclusion"],
                "Retained prepared input overlaps raw exclusions",
            )
            views, inputs = prepared_inputs(
                source, manifest_path, manifest["sources"][sid]
            )
            identities.extend(inputs)
            sources[sid] = {
                "status": "RETAINED_OPENED_INPUT_HASHES_VERIFIED",
                "trial_id": opened[sid]["trial_id"],
                "manifest": reference(manifest_path),
                "preparation": reference(preparation_path),
                "prepared_views": views,
                "unavailable_views": source["unavailable_views"],
                "original_cloud_array_sha256": opened[sid]["cloud_arrays_sha256"],
                "available_npz": inputs[0],
                "current_calibration_case": False,
            }
        runs.append(
            {
                **ref,
                "sources": len(rows),
                "measured_implementation": preparation["implementation"],
                "current_hashes_for_measured_files": current,
                "implementation_differences": differences,
                "scientific_run_repeated": False,
                "admitted_as_current_calibration_records": False,
            }
        )
    return {
        "status": "OPENED_INPUT_BINDINGS_ONLY",
        "sources": sources,
        "runs": runs,
        "unprovisioned_or_unchecked_source_ids": [
            sid for sid in opened if sid not in sources
        ],
        "input_file_references": identities,
        "calibration_cases": 0,
        "scientific_acceptance": False,
        "original_local_inputs_reverified": False,
    }
