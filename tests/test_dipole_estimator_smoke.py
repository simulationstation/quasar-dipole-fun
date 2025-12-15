import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
astropy = pytest.importorskip("astropy")
from astropy.table import Table


def test_pipeline_runs_on_small_mock(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 50
    ra = rng.uniform(0, 360, size=n)
    dec = rng.uniform(-30, 30, size=n)
    w1 = rng.uniform(10, 15, size=n)
    w1cov = rng.uniform(90, 100, size=n)

    tbl = Table()
    tbl["ra"] = ra
    tbl["dec"] = dec
    tbl["w1"] = w1
    tbl["w1cov"] = w1cov
    catalog_path = tmp_path / "mock.fits"
    tbl.write(catalog_path)

    run_tag = "smoke_test"
    cmd = [
        sys.executable,
        "pipelines/derive_catwise_constraints.py",
        "--catalog",
        str(catalog_path),
        "--run-tag",
        run_tag,
        "--bootstrap",
        "5",
        "--bmin",
        "0",
        "--apply-ecliptic-correction",
        "--ecl-model",
        "weight",
    ]
    subprocess.run(cmd, check=True)

    constraints_path = Path("results/secrest_reproduction") / f"{run_tag}" / "dipole_constraints.json"
    assert constraints_path.exists()
    data = json.loads(constraints_path.read_text())
    assert "dipole" in data
    assert "sigma_amp_used" in data
    assert "sigma_dir_deg_used" in data
