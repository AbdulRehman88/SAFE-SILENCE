# SAFE-SILENCE — Task E Real-Ocean Data Acquisition Lock

**Version:** 1.0  
**Date:** 16 September 2026  
**Scope:** OOI acquisition sources, sites, streams, variables, and downloader  
**Depends on:** Tasks B–D

## 1. Acquisition decision

Use the NSF Ocean Observatories Initiative (OOI) Coastal Endurance Array through the public OOI ERDDAP server:

> https://erddap.dataexplorer.oceanobservatories.org/erddap

ERDDAP is the primary acquisition route because it:

- is publicly accessible without M2M API credentials;
- supports reproducible variable and time subsetting;
- returns standard formats;
- exposes QARTOD aggregate quality flags;
- provides fixed dataset identifiers and REST-style URLs;
- allows independent retrieval of co-located instrument streams.

The authenticated OOI M2M route remains a fallback for calibration, deployment, annotation, or provenance products not exposed through ERDDAP. Raw environmental observations use credential-free ERDDAP access.

## 2. Primary site

### CE02SHSM — Oregon Shelf Surface Mooring

**Platform:** Near Surface Instrument Frame (RID27)  
**Coordinates in ERDDAP:** 44.63510° N, 124.30350° W  
**Common three-stream interval:** 2015-04-02 20:45 UTC to 2026-06-16 17:48 UTC

| Stream | Dataset ID | Principal measurements |
|---|---|---|
| FLORT | ooi-ce02shsm-rid27-02-flortd000 | Chlorophyll-a, optical backscatter, colored dissolved organic matter, supporting temperature/salinity |
| CTD | ooi-ce02shsm-rid27-03-ctdbpc000 | Temperature, salinity, conductivity, pressure, density |
| Dissolved oxygen | ooi-ce02shsm-rid27-04-dostad000 | Dissolved oxygen per unit mass and volume, supporting temperature/salinity/pressure |

CE02SHSM is primary because it provides the longest currently overlapping record among the selected shelf instruments and remained available through June 2026. The shelf environment is also aligned with dissolved-oxygen and biogeochemical-event monitoring.

## 3. Independent cross-site candidates

| Site | Environment | Common interval | FLORT / CTD / oxygen reference pattern |
|---|---|---|---|
| CE01ISSM | Oregon Inshore | 2014-04-17 to 2025-09-22 | RID16-02 / RID16-03 / RID16-03 |
| CE06ISSM | Washington Inshore | 2015-04-10 to 2025-07-27 | RID16-02 / RID16-03 / RID16-03 |
| CE07SHSM | Washington Shelf | 2015-04-05 to 2025-09-18 | RID27-02 / RID27-03 / RID27-04 |

Site roles and temporal grids are specified in preprocessing.yaml. Audits quantify usable coverage, natural gaps, deployment effects, event prevalence, and harmonization feasibility.

## 4. Locked environmental variables

### Primary variables

1. sea-water temperature;
2. practical salinity;
3. dissolved oxygen per unit mass;
4. chlorophyll-a concentration;
5. volume backscattering coefficient as an optical particulate/backscatter indicator.

### Supporting variables

- dissolved oxygen per unit volume;
- conductivity;
- pressure and the reported vertical coordinate;
- density;
- colored dissolved organic matter;
- latitude, longitude, and reported vertical coordinate;
- QARTOD aggregate quality flags.

### Terminology correction

The selected FLORT stream does **not** expose a variable named direct turbidity. It exposes:

> volume_backwards_scattering_coefficient_of_radiative_flux_in_sea_water

Use **optical backscatter** or a carefully defined backscatter-derived particulate proxy. It must not be called measured turbidity without an externally justified conversion.

## 5. Quality-control policy at acquisition

Task E retains all returned observations and aggregate QARTOD flags:

| Flag | Meaning |
|---:|---|
| 1 | PASS |
| 2 | NOT_EVALUATED |
| 3 | SUSPECT |
| 4 | FAIL |
| 9 | MISSING |

No observation is deleted during acquisition. The audit quantifies each flag; preprocessing.yaml specifies the cleaning policy. This preserves the ability to study genuine missingness and sensor-quality evidence without conflating it with communication loss.

## 6. Verified sampling behavior

A live retrieval test for 1 July 2024 confirmed:

- CTD: 24 records in six hours, regular 15-minute median spacing;
- FLORT: 8 records in six hours, one-minute observations within duty-cycled bursts;
- dissolved oxygen: 12 records in six hours, one-minute observations within duty-cycled bursts.

The acquisition manifest uses instrument-specific ERDDAP metadata for all 12 site-stream combinations. Dissolved-oxygen depth is optional where the stream does not expose it. Physical sensor depth is recorded through deployment metadata where available.

The downloader itself produced three raw primary-site files, three provenance sidecars, a consolidated inventory, validated row counts (8 FLORT, 24 CTD, 12 oxygen), and independent SHA-256 hashes for the six-hour test.

Consequences:

- the source data are not a continuously aligned one-minute multivariate table;
- raw streams must remain separate during acquisition;
- timestamp harmonization and aggregation frequency must be selected from the full audit rather than assumed;
- preprocessing uses the declared 15-minute and 60-minute temporal protocols;
- short within-burst dynamics and longer cross-variable dynamics must be evaluated separately.

## 7. Reproducible acquisition implementation

Files:

- ooi_acquisition_manifest.json: authoritative site, dataset, variable, range, and QC specification;
- 01_acquire_ooi_erddap.py: standard-library downloader and validator.

The downloader:

- validates every requested variable against live ERDDAP metadata;
- clips requests to each site's common availability interval;
- uses half-open, nonoverlapping time chunks;
- downloads selected variables only;
- writes raw CSV without cleaning or resampling;
- uses atomic partial-file replacement;
- retries transient network failures with exponential backoff;
- distinguishes true empty intervals from invalid HTTP 404 requests;
- validates returned headers and counts data rows;
- writes SHA-256 and exact provenance for every chunk;
- generates a consolidated acquisition inventory;
- verifies existing files before reuse;
- supports validation-only, dry-run, date-subset, site-subset, stream-subset, and forced-redownload modes.

## 8. Canonical commands

Validate identifiers and variables:

    python 01_acquire_ooi_erddap.py --validate-only

Test the primary site:

    python 01_acquire_ooi_erddap.py --site CE02SHSM --start 2024-07-01T00:00:00Z --end 2024-07-02T00:00:00Z --chunk-days 1 --output-dir data/raw/ooi

Acquire the complete primary site:

    python 01_acquire_ooi_erddap.py --site CE02SHSM --output-dir data/raw/ooi

Acquire all four sites:

    python 01_acquire_ooi_erddap.py --output-dir data/raw/ooi

The full acquisition should be executed once its persistent storage location is confirmed. Repeated runs verify and skip intact chunks.

## 9. Data-integrity boundaries

- ERDDAP times are treated as UTC.
- Availability limits are manifest snapshots and are revalidated against live metadata before downloading.
- Instrument deployment swaps are not assumed homogeneous; deployment metadata must be inspected in Task F.
- ERDDAP z is retained but cannot be assumed to equal physical sensor depth without metadata verification.
- Natural data gaps, QARTOD failures, sensor outages, and simulated acoustic outages remain distinct.
- Cross-site records are not pooled before leakage-safe split rules are locked.
- No synthetic events are introduced into the raw archive.
- Raw files are immutable inputs after checksum registration.

## 10. Acquisition provenance sources

- OOI data-access overview: https://oceanobservatories.org/data-access/
- OOI M2M interface: https://oceanobservatories.org/m2m/
- OOI ERDDAP: https://erddap.dataexplorer.oceanobservatories.org/erddap
- Coastal Endurance overview: https://ceoas.oregonstate.edu/ocean-observatories-initiative-ooi
- ERDDAP search example: https://ioos.github.io/erddapy/03-advanced_search-output.html

## 11. Acquisition checks

- the primary provider and access route are verified;
- one primary and three cross-site candidates are locked;
- exact instrument dataset IDs are recorded;
- exact variables and QC fields are recorded;
- optical backscatter is identified by its instrument variable;
- current overlapping availability ranges are recorded;
- all 12 site–instrument schemas pass live validation;
- live sample retrieval and checksum/provenance generation have been demonstrated;
- sampling mismatch is explicitly identified;
- acquisition is reproducible, resumable, checksummed, and credential-free;
- raw acquisition remains separated from cleaning, synchronization, event labeling, and simulated failures.


The audit quantifies coverage, deployments, QARTOD states, natural gaps, and sampling regimes.
The preprocessing configuration defines the modeling grids and site roles.
