# North Carolina CORA × NOAA / NDBC Validation — Streamlit

A GitHub-ready **Streamlit** application for assessing NOAA's Coastal Ocean Reanalysis (CORA-GEC) against **CO-OPS water-level gauges** and **NDBC significant wave height (Hs) buoys** for 10 North Carolina study sites.

This repository includes the original standalone batch-validation engine (`nc_cora_validation.py`) plus a browser-based interactive dashboard (`app.py`).

## Features

- The exact **10 supplied site locations**, **6 CO-OPS gauges**, and **10 NDBC buoys** are configured; optionally upload another site CSV.
- Ranked **nearest observation stations**, distances, interpretative distance flags, and an interactive station map.
- **Manual station override**, if the nearest observation is not representative.
- CORA extracted at **station coordinates**, never at the study-site coordinates; nearest native model node plus distance check.
- **CO-OPS verified hourly height** (`datum=MSL`, GMT) vs CORA `zeta` and **NDBC WVHT** vs CORA `swan_HS`.
- Strict common-hour UTC matching, with **R² (squared Pearson), NSE, RMSE, bias, coverage**, and exact overlap dates.
- Four chart groups: time series; extreme-event maximum comparison; observed vs CORA; annual maxima (timeline plus 1:1 comparison).
- Interactive multi-site and independent-station summary dashboard; download CSVs, and downloadable per-station ZIP with CSV and HTML charts.
- Optional exclusion of the two documented GEC wave issues (November 2001 / August 2004).
- The original CLI permits the full 1979–2022 validation in batch mode with figures for all unique stations and sites.

## Repository tree

```text
nc-cora-nc-streamlit/
├── app.py                        # Streamlit entry point
├── app_helpers.py                # Interactive charts + on-demand processing
├── nc_cora_validation.py         # Original full CLI processing engine
├── requirements.txt             # Python libraries for Streamlit Cloud
├── data/
│   ├── study_sites.csv           # 10 study sites (editable)
│   └── preliminary_station_assignments.csv
├── tests/test_offline.py         # Network-free automated checks
├── .streamlit/config.toml        # UI theme
├── .github/workflows/ci.yml     # GitHub Actions tests
├── .gitignore                    # Excludes caches, generated data and secrets
└── README.md
```

## 1. Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser opens the app, typically at `http://localhost:8501`.

**Typical workflow:**

1. Open **Stations & map**. Optionally refresh NOAA station metadata, check distance flags and candidate rankings.
2. In the sidebar select one site, choose water level and/or waves, and choose a CORA year range. The default is **2020–2022** to limit cloud load; full **1979–2022** is selectable.
3. Open **Run validation** and click the button. The app downloads a single observation station and the corresponding CORA native-node time series for each selected type.
4. Inspect the statistics and charts. Download a ZIP of paired data, event maxima, annual maxima and interactive charts.
5. Repeat for other stations, **import a previous result ZIP in the Overall dashboard**, or use the explicitly confirmed batch option to run all **unique nearest stations** (usually substantially fewer than 16). Open **Overall dashboard** to see the completed results mapped to the 10 sites.

**Data downloads require an internet connection.** NOAA APIs and CORA's public S3 archive are accessed live. No static or synthetic skill statistics are included.

## 2. Upload as a GitHub repository

Create a new repository in GitHub, e.g. `NC-CORA-Validation-Streamlit`. Add all files from this folder (including hidden `.streamlit` and `.github` directories), but **do not upload** `.local_data` or credentials.

Or use a terminal inside the unzipped project directory:

```bash
git init
git add .
git commit -m "Add CORA–NOAA North Carolina Streamlit validation dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/NC-CORA-Validation-Streamlit.git
git push -u origin main
```

Replace `YOUR_USERNAME` and first create the empty GitHub repository. If the repository already exists, copy these files into it and commit instead.

## 3. Deploy to Streamlit Community Cloud

1. Open <https://share.streamlit.io/> and connect your GitHub account.
2. Click **Create app** / **Deploy a public app from GitHub**.
3. Select your repository and branch (`main`). Set the **main file path** to `app.py`.
4. Choose Python **3.11/3.12** if the deployment screen offers a Python-version option.
5. Deploy. Community Cloud automatically installs `requirements.txt`.

No AWS API keys are necessary to read NOAA's **public** CORA bucket, and the CO-OPS/NDBC services used here do not require an account token. **Never commit secrets** if you add private integrations later.

**Hosting advice:** The source CORA GEC archive is about 40 TB, although the app requests only selected time series. Cloud workers may not have sufficient RAM or sustained execution time to handle a 44-year × 16-station run in one click. For the complete historic study, run the CLI on a suitably provisioned machine, zip its result folder, then import it into the Overall dashboard with matching year-range and wave-exclusion settings. App runtime caches are **ephemeral** on Streamlit Community Cloud and are not a permanent results database.

## 4. Full 1979–2022 research batch (CLI)

The original CLI is retained and accepts:

```bash
python nc_cora_validation.py --metadata-only
python nc_cora_validation.py --start-year 2019 --end-year 2022 --matched-only
python nc_cora_validation.py --start-year 1979 --end-year 2022 --matched-only
python nc_cora_validation.py --start-year 1979 --end-year 2022
```

It produces per-station and per-site CSVs and static figure files in `CORA_NC_validation_results/`. The default CLI run attempts all **16** candidate stations; `--matched-only` only processes uniquely assigned stations. Station reuse avoids pretending that shared gauge results are independent. CLI output and app output use different directory layouts, but the Streamlit import panel supports a ZIP containing the CLI output directory. For CLI archives, turn **off** the app’s wave artefact exclusion checkbox (the original CLI does not apply this exclusion).

## 5. Measurement and scientific interpretation

| Indicator | Calculation or interpretation |
| --- | --- |
| R² | Squared Pearson correlation for paired hourly values |
| NSE | `1 - sum((model - observed)^2) / sum((observed - mean(observed))^2)` |
| RMSE | `sqrt(mean((model - observed)^2))` in metres |
| Bias | `mean(model - observed)` in metres; positive means CORA overestimates |
| Coverage | Paired valid hours divided by all hours in selected CORA years |
| Event maxima | Observed 98th percentile peaks, minimum 72-hour separation, CORA maximum within ±6 h |
| Annual maxima | Model and observations both restricted to paired hours; only ≥75% coverage years shown |

**Important qualifications**

- **CORA assimilates CO-OPS observations**, so the water-level comparison is model skill/consistency assessment rather than fully independent validation.
- **MSL references/epochs may differ** between station and model; the app does not silently apply datum offsets or mean subtraction.
- The closest station by distance is not necessarily physically representative, especially for **inland sites and exposed offshore wave buoys**.
- NDBC reports may occur at e.g. `hh:50`; the previous script's standard-met parser rounds them to UTC hours. Check timestamp conventions when interpreting high-frequency errors.
- Exclusion of CORA waves for **November 2001 and August 2004** also marks the two affected years' annual wave maxima as unusable. The original CLI, without this specific switch, does not exclude them.
- The max CORA native-grid-node-to-station offset is set to **25 km** in the core. If a station is outside the wave wet-grid footprint, the analysis raises a clear error rather than extracting from an arbitrarily remote node.
- Values shown in the dashboard are live data only **after** a successful download. A shared station's scores are intentionally repeated in site-level rows; use `station_skill.csv` for independent station counts.

## 6. Site input file

`data/study_sites.csv` is included and can be edited or replaced by an uploaded CSV:

```csv
site_id,site,site_lat,site_lon
01,Town of Beaufort,34.71822,-76.663817
02,Down East Carteret County,34.75779,-76.528203
```

Site coordinates are **latitude, longitude in decimal degrees**. The app does not use them to sample CORA; they are used to find the nearest observation station.

## 7. Tests

```bash
python -m pip install pytest
python -m pytest -q
```

Offline unit tests verify matching, metrics, missing data, annual maxima, chart creation and export. They do not verify a live connection to CORA S3 or NOAA servers.

## Official sources

- NOAA CORA repository: <https://github.com/NOAA-CO-OPS/CORA-Coastal-Ocean-Reanalysis-CORA>
- NOAA CORA public data README (GEC 1979–2022, current wave revision, known issues): <https://noaa-nos-cora-pds.s3.amazonaws.com/README>
- NOAA CO-OPS API: <https://api.tidesandcurrents.noaa.gov/api/prod/>
- NDBC historical standard meteorological data: <https://www.ndbc.noaa.gov/data/historical/stdmet/>
- Streamlit Community Cloud deployment: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>

Independent research application; not affiliated with or endorsed by NOAA.
