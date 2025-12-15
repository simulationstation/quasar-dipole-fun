import json
from pathlib import Path

import pytest

try:
    import metropolis_hastings_sampler as mhs
except ImportError as exc:  # pragma: no cover - dependency missing in CI
    pytest.skip("Required scientific stack missing", allow_module_level=True)


def test_constraints_loader_updates_globals(tmp_path: Path):
    data = {
        "dipole": {"D": 0.02, "l_deg": 10.0, "b_deg": -5.0},
        "sigma_amp_used": 0.003,
        "sigma_dir_deg_used": 4.5,
    }
    path = tmp_path / "constraints.json"
    path.write_text(json.dumps(data))

    mhs.apply_constraints_from_file(str(path))

    assert mhs.D_QSO_OBS == data["dipole"]["D"]
    assert mhs.L_QSO_OBS_DEG == data["dipole"]["l_deg"]
    assert mhs.B_QSO_OBS_DEG == data["dipole"]["b_deg"]
    assert mhs.SIGMA_QSO == data["sigma_amp_used"]
    assert mhs.SIGMA_QSO_DIR_DEG == data["sigma_dir_deg_used"]
    assert mhs.CONSTRAINTS_INFO["source"] == str(path)
