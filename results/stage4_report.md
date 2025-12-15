# Stage 4: Posterior Predictive Checks in Sky Space

## Summary

This analysis tests whether the preferred models (`wall_only` vs `full`) produce distinguishable, testable predictions at the sky level, beyond fitting five summary statistics.

**Key finding**: The models produce qualitatively similar sky patterns but differ in directional stability and tracer consistency. Neither model is decisively ruled out by posterior predictive checks.

---

## 1. Posterior Predictive Sky Simulation

**Method**:
- Drew N=200 posterior samples from each model
- For each draw, computed predicted dipole vector in Galactic coordinates
- Generated mock sky with 1.4M sources and dipolar modulation P(n̂) ∝ 1 + D(n̂·d̂)
- Applied CatWISE sky mask (|b| > 30°)
- Measured dipole from simulated HEALPix map (Nside=64)

**Results stored**: `results/stage4/posterior_predictive/{model}/`

---

## 2. Directional Stability Test

| Metric | wall_only | full |
|--------|-----------|------|
| l scatter (deg) | 7.7 | 17.9 |
| b scatter (deg) | 5.4 | 13.0 |
| Mean sep from CatWISE (deg) | 33.1 | 27.1 |
| Mean sep from CMB (deg) | 40.1 | 41.9 |

**Interpretation**:
- **full shows 2.3× broader directional wandering** in l than wall_only
- This is expected: the drift component adds an extra degree of freedom that can absorb direction
- wall_only produces more stable dipole direction predictions
- Neither model's posterior centroid is particularly close to CatWISE (~30° separation)

**Result**: `results/stage4/{model}_direction_stability.json`

---

## 3. Hemispheric Asymmetry Test

| Metric | wall_only | full |
|--------|-----------|------|
| N-S asymmetry mean | -0.0049 | -0.0051 |
| N-S asymmetry std | 0.0014 | 0.0022 |
| Dipole contrast | 0.0105 | 0.0105 |

**Interpretation**:
- Both models predict essentially balanced N-S asymmetry after |b|>30° masking
- The dipole contrast (hemispheric count difference along dipole axis) is ~1% in both cases
- **No discriminating power** from hemispheric asymmetry

**Result**: `results/stage4/{model}_hemispheric_test.json`

---

## 4. Tracer-Independence Stress Test

This tests the key physical claim: **a geometric dipole should be tracer-independent**.

| Metric | wall_only | full | Expectation |
|--------|-----------|------|-------------|
| Q-R direction alignment (deg) | 8.55 ± 4.60 | 12.32 ± 10.51 | wall_only: ~0 |
| Alignment scatter (deg) | 4.60 | 10.51 | full: higher |
| Amplitude ratio Q/R | 1.37 ± 0.24 | 1.48 ± 0.36 | wall_only: ~1 |
| Ratio scatter | 0.24 | 0.36 | full: higher |

**Interpretation**:
- **wall_only shows better Q-R alignment** (8.5° vs 12.3°), but not perfect (0°)
- The non-zero wall_only alignment arises because both tracers have kinematic contributions from CMB motion (A_Q ≠ A_R), even without additional drift
- **full shows 2.3× larger alignment scatter** (10.5° vs 4.6°), indicating tracer-dependent directional variation
- Amplitude ratios differ from unity in both cases due to different response factors (A_Q/A_R)

**Key finding**: The tracer-independence test partially discriminates: wall_only shows tighter Q-R alignment, consistent with a more purely geometric origin. However, the CMB motion component still introduces tracer-dependence even in wall_only.

**Result**: `results/stage4/{model}_tracer_consistency.json`

---

## 5. Posterior Predictive Checks vs Reality

| Metric | Observed | wall_only sim | full sim | p-value (wall) | p-value (full) |
|--------|----------|---------------|----------|----------------|----------------|
| Amplitude D | 0.0154 | 0.0154 | 0.0154 | 0.460 | 0.480 |
| CMB separation (deg) | 28.8 | 40.1 | 41.9 | 0.155 | 0.355 |
| Within 10° of CatWISE | - | 0.0% | 1.5% | - | - |

**Interpretation**:
- **Amplitude p-values are centered** (~0.5): both models predict the observed amplitude well
- **CMB separation p-values are non-extreme**: the observed ~29° separation from CMB is within the posterior predictive distribution
- **Neither model strongly favors direction recovery** within 10° of CatWISE (0-1.5%)

**Result**: `results/stage4/{model}_ppc.json`

---

## Physical Conclusion

**Do sky-level predictions favor a geometric dipole over observer motion?**

The evidence is **weakly suggestive but not conclusive**:

1. **Direction stability**: wall_only shows 2.3× tighter directional scatter than full, consistent with a fixed geometric origin
2. **Tracer consistency**: wall_only shows better Q-R alignment (8.5° vs 12.3°), though not perfect alignment (0°) due to CMB kinematic component
3. **Amplitude recovery**: Both models reproduce the observed amplitude equally well
4. **Overall fit**: Neither model is ruled out by posterior predictive p-values

The dipole is consistent with being **primarily geometric** (structure-induced), with a **small kinematic contamination** from our motion relative to the CMB. The full model's additional drift component does not significantly improve the fit but introduces extra directional scatter.

---

## Falsification Criteria

**What would falsify the geometric (wall) model**:

1. Detection of a dipole in a tracer with A ~ 0 (e.g., a sample with known zero magnification response)
2. Measurement of dipole direction that rotates systematically with redshift
3. Observation of dipole direction aligned with CMB to within <5° in multiple independent catalogs
4. Detection of frequency-dependent dipole direction in radio continuum surveys

**What would falsify the kinematic (drift) model**:

1. Measurement of dipole direction >45° from any kinematic prediction
2. Detection of identical dipole direction in tracers with vastly different spectral indices (expecting tracer-dependent kinematic component)
3. Null detection of kinematic dipole in CMB spectral distortions

---

## Future Decisive Data

**Data that would decisively settle the question**:

1. **Euclid/Roman quasar dipole**: Independent catalog with different selection effects; geometric dipole predicts same direction
2. **Peculiar velocity surveys**: Direct measurement of bulk flow within 300 Mpc would constrain drift component
3. **Radio dipole at multiple frequencies**: If geometric, direction should be frequency-independent
4. **CMB spectral distortion measurement**: Would directly measure kinematic component independent of number counts

---

## Output Files

```
results/stage4/
├── wall_only_samples.csv           # Posterior samples
├── wall_only_direction_stability.json
├── wall_only_hemispheric_test.json
├── wall_only_tracer_consistency.json
├── wall_only_ppc.json
├── full_samples.csv
├── full_direction_stability.json
├── full_hemispheric_test.json
├── full_tracer_consistency.json
├── full_ppc.json
└── stage4_combined.json
```

---

*Report generated: 2025-12-14*
*Pipeline: run_stage4_ppc.py*
