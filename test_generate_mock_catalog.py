"""Lightweight tests for generate_mock_catalog utilities."""

import numpy as np

from generate_mock_catalog import unitvec_to_radec


def test_unitvec_to_radec_axes():
    ra_x, dec_x = unitvec_to_radec((1, 0, 0))
    assert np.isclose(dec_x, 0.0, atol=1e-8)
    assert np.isclose(ra_x % 360.0, 0.0, atol=1e-8)

    ra_y, dec_y = unitvec_to_radec((0, 1, 0))
    assert np.isclose(dec_y, 0.0, atol=1e-8)
    assert np.isclose(ra_y % 360.0, 90.0, atol=1e-8)

    ra_z, dec_z = unitvec_to_radec((0, 0, 1))
    assert np.isclose(dec_z, 90.0, atol=1e-8)
    # RA is undefined at the poles; ensure it is finite
    assert np.isfinite(ra_z)

    ra_neg_z, dec_neg_z = unitvec_to_radec((0, 0, -1))
    assert np.isclose(dec_neg_z, -90.0, atol=1e-8)
    assert np.isfinite(ra_neg_z)

