# README: Relevant Data Points for ML Model from Kepler, K2, and TESS Exoplanet Archives

This document summarizes the key data columns from NASA Exoplanet Archive catalogs for **Kepler**, **K2**, and **TESS (TOI)** missions, which should be consolidated and used as inputs/features for the Machine Learning (ML) model aimed at exoplanet detection and characterization.

---

## Data Sources

- **Kepler Cumulative Catalog:**  
  [Kepler Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

- **K2 Planet and Candidate Catalog:**  
  [K2 Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)

- **TESS Object of Interest (TOI) Catalog:**  
  [TESS TOI Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)

---

## Recommended Columns to Use (Common Across Missions)

| Feature              | Description                                           | Notes                                      |
|----------------------|-------------------------------------------------------|--------------------------------------------|
| **Planet parameters:** |                                                       |                                            |
| `pl_orbper`          | Orbital period (days)                                 | Periodicity of the planet’s orbit          |
| `pl_tranmid` (or equivalent) | Transit mid-time (epoch, in BJD)                      | Center of transit for phase folding        |
| `pl_trandurh`        | Transit duration (hours)                              | Duration of transit event                   |
| `pl_trandep`         | Transit depth (fractional flux decrease)             | Relative drop in stellar brightness        |
| `pl_rade`            | Planet radius (in Earth radii)                        | Key planetary property                      |
| **Stellar parameters:** |                                                       |                                            |
| `st_teff`            | Stellar effective temperature (K)                     | Affects limb darkening and transit depth   |
| `st_rad`             | Stellar radius (in Solar radii)                       | Used to scale planetary size and transit   |
| `st_logg`            | Stellar surface gravity                                | Useful for stellar characterization         |
| `st_dist`            | Stellar distance (parsecs)                            | Can be helpful for flux calibration        |
| **Transit signal proxies:** |                                                   |                                            |
| `pl_insol`           | Insolation flux (relative to Earth)                   | Environmental parameter                      |
| **Identification & labels:** |                                                 |                                            |
| `kepid` / `tid` / `epic_hostname` | Unique star identifiers                              | Required for cross-matching and queries    |
| `disposition`        | Planet status (CONFIRMED, CANDIDATE, FALSE POSITIVE)  | Target label for supervised learning       |

---

## Notes and Consolidation Guidelines

- Column names might differ slightly across catalogs:  
  - Kepler uses `kepid`; TESS uses `tid`; K2 uses `epic_hostname`. These should be harmonized as a general `"id"` or `"name"` column.  
  - Transit mid-time may be `pl_tranmid` or similarly named.  
- Not all catalogs provide all parameters; use available columns and handle missing values cautiously.  
- Common columns around orbital and transit parameters, stellar properties, and disposition are most crucial for your ML model.  
- Use these stellar and planetary parameters as **metadata features** to augment the time series input from light curves.

---

## Using Lightkurve-derived Light Curve Data

- For ML input time series, use **PDC (Pre-search Data Conditioning) light curves**, which are cleaned flux time series optimal for transit detection.  
- Phase fold light curves using `pl_orbper` and `pl_tranmid` to align transit events.  
- Use the `pdcsap_flux` field in Lightkurve objects for flux time series input.

---

## Summary of Final Input Features for ML Model

- **Time Series Input:** Phase folded, normalized PDC light curve flux vs time segments around transits.  
- **Tabular Features:**
  - Orbital period (`pl_orbper`)  
  - Transit mid-time (`pl_tranmid`)  
  - Transit duration (`pl_trandurh`)  
  - Transit depth (`pl_trandep`)  
  - Planet radius (`pl_rade`)  
  - Stellar effective temperature (`st_teff`)  
  - Stellar radius (`st_rad`)  
  - Stellar surface gravity (`st_logg`)  
  - Stellar distance (`st_dist`)  
  - Planet disposition/class label (`disposition`)  

---

For more details, refer to the original NASA Exoplanet Archive catalogs and Lightkurve documentation.

---

*Prepared for ML-based exoplanet detection and classification pipeline development.*
