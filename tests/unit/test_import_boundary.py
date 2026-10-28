"""U-import-boundary: the label-independence rules of docs/6 section 4.3, checked on the import graph.

- `uq` never imports `truth`, directly or transitively;
- `sdr` (receiver running) imports neither `truth` nor `uq`, except the loop model in `uq.loop` for the inverse map
  (D-21; importing it loads the `gnssaudit.uq` package itself);
- `inject` (disturbance synthesis) imports neither `truth` nor `uq`;
- `screen` (segment admission) and `io.fairdata` (prefix streaming) import neither `truth` nor `uq` (P2b).
"""

import subprocess
import sys

import pytest

PROBE = """
import importlib, pkgutil, sys
pkg = importlib.import_module("{package}")
for m in pkgutil.walk_packages(getattr(pkg, "__path__", []), "{package}."):  # a plain module has no __path__
    importlib.import_module(m.name)
forbidden, allowed = {forbidden!r}, {allowed!r}
bad = sorted(n for n in sys.modules if n not in allowed and any(n == f or n.startswith(f + ".") for f in forbidden))
print(",".join(bad))
"""


SDR_MAY_IMPORT = ["gnssaudit.uq", "gnssaudit.uq.loop"]  # D-21: the loop model only (final review R2, minor 9)


@pytest.mark.parametrize(
    ("package", "forbidden", "allowed"),
    [
        ("gnssaudit.uq", ["gnssaudit.truth"], []),
        ("gnssaudit.sdr", ["gnssaudit.truth", "gnssaudit.uq"], SDR_MAY_IMPORT),
        ("gnssaudit.inject", ["gnssaudit.truth", "gnssaudit.uq"], []),
        ("gnssaudit.inject.realif", ["gnssaudit.truth", "gnssaudit.uq"], []),
        ("gnssaudit.screen", ["gnssaudit.truth", "gnssaudit.uq"], []),
        ("gnssaudit.io.fairdata", ["gnssaudit.truth", "gnssaudit.uq"], []),
    ],
)
def test_package_does_not_import_forbidden_packages(package: str, forbidden: list[str], allowed: list[str]) -> None:
    probe = PROBE.format(package=package, forbidden=forbidden, allowed=allowed)
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""
