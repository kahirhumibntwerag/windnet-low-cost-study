"""Retrieve scalar SDO/EVE tables only; join them to isolated training rows.

Daily products are retrospective calibrated measurements, not an as-issued
operational archive. No image or solar-wind evaluation target is downloaded.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import h5py
from astropy.time import Time, TimeDelta
from prepare_training import load_training_bundle, EXPORT_DIRECTORY, sha256

SOURCE = "https://lasp.colorado.edu/eve/data_access/eve_data/products/merged/latest_EVE_L3_merged_1nm.ncdf"
LINES = [("Fe_IX_17", "Fe IX", 17.1073), ("Fe_XII_20", "Fe XII", 19.512),
         ("Fe_XIV_21", "Fe XIV", 21.1331), ("He_II_30", "He II", 30.3783),
         ("Fe_XVI_34", "Fe XVI", 33.541), ("He_I_58", "He I", 58.4334)]
LAGS = (5, 6, 7, 8)


def fetch() -> dict:
    b = load_training_bundle()
    root = EXPORT_DIRECTORY
    cache = root / "eve_source"
    cache.mkdir(exist_ok=True)
    dates = b["date"].astype("datetime64[D]")
    start, stop = str(dates.min() - np.timedelta64(8, "D")), str(dates.max() - np.timedelta64(4, "D"))
    raw = cache / "latest_EVE_L3_merged_1nm.ncdf"
    if not raw.exists():
        response = requests.get(SOURCE, timeout=(15, 60))
        response.raise_for_status()
        raw.write_bytes(response.content)
    fields, values, channel_metadata = [], [], []
    with h5py.File(raw, "r") as f:
        if f.attrs['product_version'].decode() != '008':
            raise ValueError("Expected the frozen EVE V8 calibration")
        # Use explicit metadata identification, not HAPI's stale line ordering.
        waves = f['linesmeta_wave_center'][:]
        ions = [s.decode().strip() for s in f['linesmeta_name'][:]]
        for name, ion, wave in LINES:
            hits = [i for i, (a, c) in enumerate(zip(ions, waves)) if a == ion and abs(c-wave) < .002]
            if len(hits) != 1:
                raise ValueError(f"Ambiguous line identity: {name}")
            i = hits[0]
            fields.append(name); values.append(f['line_irradiance'][i, :])
            channel_metadata.append({'name': name, 'ion': ion, 'wavelength_nm': float(waves[i]), 'units': 'W/m^2'})
        for name, i in (("diode171", 1), ("diode257", 2), ("diode304", 3)):
            description = f['diodemeta_name'][i].decode().strip()
            fields.append(name); values.append(f['diode_irradiance'][i, :])
            channel_metadata.append({'name': name, 'description': description, 'units': 'W/m^2'})
        tai = f['time'][:]
        source_created = str(f.attrs['date_created'][0])
        # Convert TAI with leap seconds; cross-check independent year/day field.
        time = Time("1958-01-01T00:00:00", scale="tai") + TimeDelta(tai, format="sec")
        utc_days = np.asarray(time.utc.isot, dtype='datetime64[ms]').astype('datetime64[D]')
        yday = pd.to_datetime(f['yyyydoy'][:].astype(int).astype(str), format='%Y%j').to_numpy().astype('datetime64[D]')
        if not np.array_equal(utc_days, yday):
            raise ValueError('Independent EVE time fields disagree')
    daily = pd.DataFrame(np.column_stack(values), columns=fields, index=utc_days)
    daily = daily.loc[(daily.index >= np.datetime64(start)) & (daily.index < np.datetime64(stop))]
    daily = daily.where(np.isfinite(daily) & (daily > 0))
    if daily.index.duplicated().any():
        raise ValueError('Duplicate EVE daily times')
    sources = [{'url': SOURCE, 'sha256': sha256(raw), 'download_bytes': raw.stat().st_size,
                'version': '008', 'source_created': source_created, 'channels': channel_metadata}]
    names, columns, provenance = [], [], []
    logged = []
    # Source-fold IDs for calendar lags 5..8 are already in the isolated wind
    # provenance. Apply the same strict excluded-fold guard to solar features.
    for lag in LAGS:
        source_fold = b["source_fold_recurrence"][:, 4 + lag - 3]
        value = daily.reindex(dates - np.timedelta64(lag, "D")).to_numpy(float)
        value[source_fold == 5] = np.nan
        value = np.log(value)
        logged.append(value)
        for j, field in enumerate(daily.columns):
            names.append(f"log_{field}_lag{lag}")
            columns.append(value[:, j])
            provenance.append(source_fold)
    # Summaries require all four days, so their provenance is a four-way union.
    stack = np.stack(logged, axis=1)
    for label, matrix in (("mean", np.mean(stack, axis=1)), ("trend", (stack[:, 0] - stack[:, -1]) / 3)):
        for j, field in enumerate(daily.columns):
            names.append(f"log_{field}_{label}_lags5to8")
            columns.append(matrix[:, j])
            provenance.append(np.zeros(len(dates), dtype=np.int8))
    for a, c in (("Fe_XII_20", "Fe_XIV_21"), ("Fe_IX_17", "Fe_XVI_34")):
        ia, ic = list(daily.columns).index(a), list(daily.columns).index(c)
        for k, lag in enumerate(LAGS):
            names.append(f"logratio_{a}_{c}_lag{lag}")
            columns.append(stack[:, k, ia] - stack[:, k, ic])
            provenance.append(b["source_fold_recurrence"][:, 4 + lag - 3])
    x = np.column_stack(columns)
    # A per-row source matrix lets the trainer additionally mask inner-fold
    # values before fitting. Mean/trend are recomputed after that inner mask.
    np.savez_compressed(root / "eve_features.npz", rowid=b["rowid"], eve=x,
                        lag_values=stack, lag_source_fold=np.column_stack([b["source_fold_recurrence"][:, 4 + lag - 3] for lag in LAGS]),
                        names=np.asarray(names, dtype='U'), fields=np.asarray(daily.columns, dtype='U'))
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "feature_bundle_sha256": sha256(root / "eve_features.npz"),
        "train_only_sha256": sha256(root / "train_only.npz"),
        "lags_calendar_days": list(LAGS), "scalar_channels": len(daily.columns),
        "feature_count": x.shape[1], "rows": len(dates),
        "finite_fraction_by_feature_training_only": {name: float(np.isfinite(x[:, j]).mean()) for j, name in enumerate(names)},
        "availability_rule": "Completed daily observations precede nominal t-3 issue by >=2 days; lags t-5 through t-8. Retrospective V8 calibration does not establish as-issued availability.",
        "prospective_channel_exclusions": {"diode1216": "Fourier-filter temporal support undocumented; excluded before model execution.", "diode366": "Official ESP documentation reports unstable shunt resistance; excluded before model execution."},
        "limitations": ["Present-day calibrated science product, not historical as-issued data.", "MEGS-A failure 2014-05-26 removes short wavelength lines; no gap interpolation.", "Full-disk irradiance loses spatial coronal-hole geometry; absence of gain would not refute spatial AIA features."],
        "source_readme": "https://lasp.colorado.edu/eve/data_access/eve_data/products/level3/EVE_L3_V8_README.pdf",
        "raw_images_downloaded": 0,
    }
    (root / "eve_features.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = fetch()
    print(json.dumps({k: m[k] for k in ("rows", "scalar_channels", "feature_count", "feature_bundle_sha256", "raw_images_downloaded")}))
