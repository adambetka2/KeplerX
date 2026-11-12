"""Prepare locally versioned dependency repairs from authenticated releases."""

import argparse
import subprocess
import tarfile
import zipfile
from pathlib import Path

from post_boi.records import digest, read, receipt, require, write

RELEASES = {
    "batman-source.tar.gz": "fbe07e6fa22208a5aa38607b73dc828563cb7496624848b488d587056262f392",
    "juliet-2.2.10-py3-none-any.whl": "a2ee7dc97c1ad71d6cb9b52742774606b9f8e57fdae419724678a0574a2e0e7f",
}


def quarter_patch_state(patched):
    import triceratops

    root = Path(triceratops.__file__).parent.parent
    manifest = read(Path(__file__).with_name("patches") / "triceratops-quarters.json")
    key = "patched_sha256" if patched else "original_sha256"
    for name, checksums in manifest["files"].items():
        require(
            digest(root / name) == checksums[key],
            "Quarter likelihood source differs: " + name,
        )
    return root


def patch_quarter_runtime():
    """Apply only to a new, unused runtime containing the pinned upstream files."""
    root = quarter_patch_state(patched=False)
    apply_patch(root, "triceratops-quarters.patch")
    quarter_patch_state(patched=True)


def apply_patch(root, name):
    patch = Path(__file__).with_name("patches") / name
    subprocess.run(
        ["patch", "--batch", "--forward", "-p1", "-i", str(patch.resolve())],
        cwd=root,
        check=True,
    )
    return receipt(patch)


def prepare(sources, output):
    sources, output = Path(sources).resolve(), Path(output).resolve()
    for name, expected in RELEASES.items():
        require(digest(sources / name) == expected, "Upstream archive differs: " + name)
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(sources / "batman-source.tar.gz") as archive:
        archive.extractall(output, filter="data")
    batman = output / "batman-6fc570e209fa2e3dadd2ffa4c535f528a7bef495"
    batman.rename(output / "batman")
    juliet = output / "juliet"
    with zipfile.ZipFile(sources / "juliet-2.2.10-py3-none-any.whl") as archive:
        archive.extractall(juliet)
    patches = [apply_patch(output / "batman", "batman-2.5.3.patch")]
    patches.append(apply_patch(juliet, "juliet-2.2.10.patch"))
    info = juliet / "juliet-2.2.10.dist-info"
    metadata = info / "METADATA"
    metadata.write_text(
        metadata.read_text().replace(
            "Version: 2.2.10\n", "Version: 2.2.10+spacelab.1\n"
        )
    )
    info.rename(juliet / "juliet-2.2.10+spacelab.1.dist-info")
    write(
        output / "sources.json",
        dict(
            upstream=[receipt(sources / name) for name in RELEASES],
            patches=patches,
            scope="Numerical/dependency repairs only; no altered transit physics, eccentricity distribution or TRICERATOPS source.",
        ),
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.sources, args.output)
