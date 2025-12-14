# Catalog schema for dipole estimation

This pipeline works directly on local sky catalogs. It never attempts to download data and refuses to proceed if required columns are missing. The same schema is expected for CSV or Parquet input.

## Required columns

| Column | Description | Units / frame |
| --- | --- | --- |
| `ra` **or** `lon` | Right ascension in degrees (ICRS/J2000) or sky longitude used for the dipole fit. If `lon`/`lat` are provided, they are treated as the working coordinate frame and no conversion is attempted. | degrees |
| `dec` **or** `lat` | Declination in degrees (ICRS/J2000) or sky latitude used for the dipole fit. | degrees |

### Coordinate conventions

- If you only provide `ra`/`dec`, the pipeline internally converts to Galactic coordinates for masking and outputs the dipole direction in that Galactic frame.
- If you supply `lon`/`lat`, the pipeline assumes these are already in the frame you want for the dipole (e.g., Galactic). Make sure the frame matches any magnitude or mask choices you intend to make.

## Optional columns

| Column | Purpose |
| --- | --- |
| `weight` | Per-source weight (e.g., completeness correction). If omitted, unit weights are used. |
| `mag` | Apparent magnitude used for selection (e.g., `w1mag` from CatWISE). Names are user-configurable through CLI flags. |
| `mask_flag` | Boolean column where `True`/`1` marks sources to **keep**. This allows passing external sky masks. |

## Magnitude columns

Magnitude columns are never assumed. Provide the name through `--mag-col`. Magnitudes are treated as scalar values; the pipeline applies inclusive limits `mag_min <= mag <= mag_max` when both bounds are supplied.

## Masking flags

- Galactic latitude cuts are applied using either supplied Galactic `lat` or derived from `ra`/`dec`.
- Rectangular masks can be passed via CLI as longitude/latitude ranges.
- A user-defined boolean column can be used to drop objects when the value is `False`.

## Example rows

CSV example (header shown; rows truncated):

```csv
ra,dec,w1mag,weight,clean_mask
150.1234,2.3456,15.8,1.0,True
151.9876,-1.2345,16.2,0.8,True
```

Parquet example (schema description):

- `ra` (double)
- `dec` (double)
- `w1mag` (float)
- `weight` (float)
- `clean_mask` (boolean)

## Validation checklist before running

1. Confirm angle columns are in degrees.
2. Ensure there are no NaN values in the angle columns after applying quality cuts.
3. Verify the chosen magnitude column exists and is finite for all rows you plan to keep.
4. Decide whether weights should reflect completeness or flux; set to unity if unsure.
5. For partial-sky catalogs, provide a mask column to avoid biased dipole estimates.

The estimator intentionally fails fast if any required column is missing or if fewer than 100 sources remain after cuts.
