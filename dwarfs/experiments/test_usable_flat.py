"""Year-null availability must follow the selected photometry product."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import usable_flat as usable


def test_old_full_psf_scramble_failure_does_not_exclude_flat_candidate():
    item = {
        "case": {"kind": "year_scrambled", "preparation_status": "SCRAMBLE_INELIGIBLE"},
        "flat_cadences": 40000,
        "quarter_count": 18,
    }
    assert usable.potential(item)
    assert not usable.potential(item | {"quarter_count": 16})
    assert not usable.potential(item | {"flat_cadences": 0})
