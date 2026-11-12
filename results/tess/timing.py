"""Standard TESS spacecraft timing with a frozen, external SPICE cache."""

import json
from pathlib import Path

from astropy.config.paths import set_temp_cache
from astropy.utils.data import cache_contents, conf

from tess.archive import digest, write


def spacecraft(cache):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = cache / "kernels.json"
    with set_temp_cache(cache), conf.set_temp("remote_timeout", 120):
        import lkspacecraft

        if manifest.exists():
            files = cache_contents(pkgname="lkspacecraft")
            for url, checksum in json.loads(manifest.read_text()).items():
                if url not in files or digest(files[url]) != checksum:
                    raise ValueError("SPICE cache differs from its saved manifest")
            lkspacecraft.enable_offline_mode()
        else:
            lkspacecraft.disable_offline_mode()
        result = lkspacecraft.TESSSpacecraft()
        if not manifest.exists():
            write(
                manifest,
                {
                    url: digest(path)
                    for url, path in cache_contents(pkgname="lkspacecraft").items()
                    if "TestMeta" not in url
                },
            )
    return result, digest(manifest)
