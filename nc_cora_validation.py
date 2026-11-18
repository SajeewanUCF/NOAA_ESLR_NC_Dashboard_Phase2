#!/usr/bin/env python3
"""CORA-GEC (1979-2022) versus NOAA CO-OPS and NDBC observations.

For each of 10 North Carolina sites, find the nearest station *among the 6/10
user-supplied candidate stations*. Sample the CORA native grid at the selected
observation station's coordinates, never at the study site's coordinates.

Outputs include site assignments, aligned observations, paired-hour skill,
annual maxima, independent observed events, 4-panel technical figures,
individual figures, a summary dashboard and an HTML map.

CORA source: https://github.com/NOAA-CO-OPS/CORA-Coastal-Ocean-Reanalysis-CORA
Dataset: https://noaa-nos-cora-pds.s3.amazonaws.com/README

Usage:
  python nc_cora_validation.py --metadata-only
  python nc_cora_validation.py                  # complete available window
  python nc_cora_validation.py --start-year 2020 --end-year 2022
  python nc_cora_validation.py --matched-only       # faster, only matched gauges

Requires Internet connectivity at runtime; this script never downloads the
multi-terabyte CORA archive, but retrieves only selected point time series.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
import math
import re
import shutil
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

# -------------------------- CONFIGURATION ----------------------------------
SITES = [
    ('01', 'Town of Beaufort',          34.71822, -76.663817),
    ('02', 'Down East Carteret County', 34.75779, -76.528203),
    ('03', 'Town of Kitty Hawk',        36.06461, -75.705733),
    ('04', 'Cape Carteret',             34.69155, -77.063003),
    ('05', 'Currituck County',          36.41898, -75.946322),
    ('06', 'Town of Swansboro',         34.68766, -77.119114),
    ('07', 'Town of Washington Park',   35.53266, -77.032450),
    ('08', 'Town of Windsor',           35.99849, -76.946064),
    ('09', 'Town of Duck',              36.16503, -75.753558),
    ('10', 'Town of Aurora',            35.30544, -76.788542),
]
COOPS = {
    '8651370': 'Duck, NC',
    '8652587': 'Oregon Inlet Marina, NC',
    '8654467': 'USCG Station Hatteras, NC',
    '8656483': 'Beaufort, Duke Marine Lab, NC',
    '8658163': 'Wrightsville Beach, NC',
    '8658120': 'Wilmington, NC',
}
# NOAA NDBC station_table.txt coordinates, retrieved Nov 2026. The program
# refreshes these automatically at runtime and retains these as fallback.
# Published NOAA CO-OPS coordinates (fallback only; refreshed from MDAPI).
COOPS_COORDS_FALLBACK = {
    '8651370': (36.1833, -75.7467),
    '8652587': (35.7957, -75.5482),
    '8654467': (35.2086, -75.7042),
    '8656483': (34.7173, -76.6707),
    '8658163': (34.21333, -77.78667),
    '8658120': (34.2275, -77.95361),
}
NDBC = {
    '41108': ('Wilmington Harbor, NC',        33.719, -78.016),
    '41110': ('Masonboro Inlet, NC',          34.144, -77.715),
    '41159': ('Onslow Bay Outer, NC',         34.211, -76.944),
    '41120': ('Cape Hatteras East, NC',       35.258, -75.285),
    '44095': ('Oregon Inlet, NC',             35.750, -75.330),
    '44086': ('Nags Head, NC',                36.001, -75.421),
    '44056': ('Duck FRF, NC',                 36.200, -75.714),
    '44100': ('Duck FRF 26m, NC',             36.257, -75.593),
    '44088': ('Virginia Beach Offshore, VA',  36.612, -74.839),
    '44014': ('Virginia Beach 64NM East, VA', 36.603, -74.837),
}
GEC_FIRST, GEC_LAST = 1979, 2022
MAX_NODE_DISTANCE_KM = 25.0
# Distance flags are screening heuristics, not hydrodynamic suitability tests.
SITE_DISTANCE_FLAG_KM = {'WL': 30.0, 'HS': 50.0}
MIN_YEAR_COVERAGE = 0.75   # paired hourly records / all hours of year
EVENT_THRESHOLD_QUANTILE = 0.98
EVENT_DECLUSTER_HOURS = 72
EVENT_MATCH_WINDOW_HOURS = 6
EVENTS_TO_PLOT = 12
COOPS_URL = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter'
COOPS_META = 'https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{}.json'
NDBC_TABLE_URL = 'https://www.ndbc.noaa.gov/data/stations/station_table.txt'
NDBC_STD_URL = 'https://www.ndbc.noaa.gov/data/historical/stdmet/{}h{}.txt.gz'
CORA_S3 = 's3://noaa-nos-cora-pds/'
CORA_CATALOG = {
    'WL': 'CORA-GEC-fort.63-timeseries',
    'HS': 'CORA-GEC-swan_HS.63-timeseries',
}
CORA_VARIABLE = {'WL': 'zeta', 'HS': 'swan_HS'}
LABEL = {'WL': 'Water level (m, MSL)', 'HS': 'Significant wave height Hs (m)'}
OBS_COLOR, MODEL_COLOR = '#172c45', '#dd6b2e'

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
LOG = logging.getLogger('CORA-NC')


def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')


def session_with_retries():
    ses = requests.Session()
    retry = Retry(total=3, read=3, connect=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504])
    ses.mount('https://', HTTPAdapter(max_retries=retry, pool_connections=8))
    ses.headers.update({'User-Agent': 'NC-CORA-validation-research/1.0'})
    return ses


def haversine_km(lat0, lon0, lat1, lon1):
    """Great-circle distance, accepts scalar or arrays."""
    a0, o0, a1, o1 = np.deg2rad([lat0, lon0, lat1, lon1])
    aa = np.sin((a1 - a0) / 2)**2 + np.cos(a0)*np.cos(a1)*np.sin((o1-o0)/2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(aa, 0, 1)))


def unit_sphere(lat, lon):
    a, o = np.deg2rad(np.asarray(lat)), np.deg2rad(np.asarray(lon))
    return np.column_stack([np.cos(a)*np.cos(o), np.cos(a)*np.sin(o), np.sin(a)])


def cached_request_bytes(sess, url, cache_path):
    """Do not cache transient failures. Cache confirmed HTTP-404 as empty bytes."""
    if cache_path.exists():
        return cache_path.read_bytes()
    try:
        r = sess.get(url, timeout=65)
        if r.status_code == 404:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b'')
            return b''
        r.raise_for_status()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(r.content)
        return r.content
    except requests.RequestException as exc:
        LOG.warning('Could not retrieve %s: %s', url, exc)
        return None


def get_coops_metadata(sess, station_id, cache_dir):
    f = cache_dir / 'metadata' / f'coops_{station_id}.json'
    raw = cached_request_bytes(sess, COOPS_META.format(station_id), f)
    record = None
    if raw:
        try:
            info = json.loads(raw)
            if isinstance(info, dict) and 'stations' in info:
                info = info['stations']
            record = info[0] if isinstance(info, list) and info else info
            lat = float(record.get('lat', record.get('latitude')))
            lon = float(record.get('lng', record.get('lon', record.get('longitude'))))
            return {'id': station_id, 'name': COOPS[station_id],
                    'lat': lat, 'lon': lon, 'type': 'WL'}
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            LOG.warning('Unexpected metadata for CO-OPS %s: %s', station_id, exc)
    # API datagetter metadata fallback; one month that may be within record
    for y in (2021, 2019, 2015, 2010, 2000):
        try:
            resp = sess.get(COOPS_URL, params=dict(begin_date=f'{y}0101',
                end_date=f'{y}0131', station=station_id, product='hourly_height',
                datum='MSL', time_zone='gmt', units='metric', format='json'),
                timeout=40)
            j = resp.json()
            if 'metadata' in j:
                return {'id': station_id, 'name': COOPS[station_id],
                        'lat': float(j['metadata']['lat']),
                        'lon': float(j['metadata']['lon']), 'type': 'WL'}
        except (requests.RequestException, ValueError, KeyError, TypeError):
            continue
    if station_id in COOPS_COORDS_FALLBACK:
        LOG.warning('Using previously published NOAA coordinates for %s', station_id)
        lat, lon = COOPS_COORDS_FALLBACK[station_id]
        return {'id': station_id, 'name': COOPS[station_id],
                'lat': lat, 'lon': lon, 'type': 'WL'}
    raise RuntimeError(f'No verified coordinates for CO-OPS station {station_id}')


def get_ndbc_metadata(sess, cache_dir):
    result = {code: dict(id=code, name=x[0], lat=x[1], lon=x[2], type='HS')
              for code, x in NDBC.items()}
    raw = cached_request_bytes(sess, NDBC_TABLE_URL,
                               cache_dir/'metadata'/'ndbc_station_table.txt')
    if not raw:
        LOG.warning('Using verified fallback NDBC coordinates')
        return result
    for line in raw.decode('utf-8', errors='replace').splitlines():
        parts = line.split('|')
        if len(parts) < 7:
            continue
        code = parts[0].strip()
        if code not in result:
            continue
        m = re.search(r'([\d.]+)\s*([NS])\s+([\d.]+)\s*([EW])', parts[6])
        if m:
            lat = float(m[1]) * (1 if m[2] == 'N' else -1)
            lon = float(m[3]) * (1 if m[4] == 'E' else -1)
            result[code]['lat'], result[code]['lon'] = lat, lon
            result[code]['name'] = parts[4].strip() or result[code]['name']
    return result


def assign_stations(sites, stations):
    mapping, candidates = [], []
    for ident, name, lat, lon in sites:
        entry = dict(site_id=ident, site=name, site_lat=lat, site_lon=lon)
        for typ in ['WL', 'HS']:
            distances = sorted(
                [(float(haversine_km(lat, lon, p['lat'], p['lon'])), p)
                 for p in stations.values() if p['type'] == typ],
                key=lambda item: item[0])
            best_dist, best = distances[0]
            entry.update({f'{typ}_station_id': best['id'],
                          f'{typ}_station_name': best['name'],
                          f'{typ}_distance_km': round(best_dist, 2),
                          f'{typ}_distance_flag': bool(best_dist > SITE_DISTANCE_FLAG_KM[typ]),
                          f'{typ}_station_lat': best['lat'],
                          f'{typ}_station_lon': best['lon']})
            for rank, (distance, sta) in enumerate(distances, 1):
                candidates.append(dict(site_id=ident, site=name, variable=typ,
                                       rank=rank, station_id=sta['id'],
                                       station=sta['name'], distance_km=round(distance, 3)))
        mapping.append(entry)
    return pd.DataFrame(mapping), pd.DataFrame(candidates)


def coops_year(sess, station_id, year, cache_dir):
    """Verified hourly heights, MSL, GMT. Half-year calls stay within NOAA limits."""
    cache_path = cache_dir / 'coops_parsed' / f'{station_id}_{year}.csv'
    if cache_path.exists():
        return read_ts_csv(cache_path)
    frames = []
    successful = True
    for begin, end in [(f'{year}0101', f'{year}0630'),
                       (f'{year}0701', f'{year}1231')]:
        params = dict(begin_date=begin, end_date=end, station=station_id,
                      product='hourly_height', datum='MSL', time_zone='gmt',
                      units='metric', format='json', application='CORA_NC_Validation')
        try:
            resp = sess.get(COOPS_URL, params=params, timeout=70)
            resp.raise_for_status()
            payload = resp.json()
            if 'data' in payload:
                fr = pd.DataFrame(payload['data'])
                if not fr.empty:
                    fr['time'] = pd.to_datetime(fr['t'], utc=True, errors='coerce')
                    fr['value'] = pd.to_numeric(fr['v'], errors='coerce')
                    frames.append(fr[['time', 'value']])
            elif 'error' in payload:
                LOG.debug('CO-OPS %s %s-%s: %s', station_id, begin, end, payload['error'])
        except (requests.RequestException, ValueError) as exc:
            successful = False
            LOG.warning('CO-OPS %s %s-%s failed: %s', station_id, begin, end, exc)
    if not successful:
        return pd.Series(dtype=float, name='observed')
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=['time','value'])
    series = clean_timeseries(df, 'observed', low=-20, high=20)
    write_ts_csv(series, cache_path)
    return series


def parse_ndbc_stdmet(raw, year):
    """Read an annual NDBC stdmet text/gzip table, retaining WVHT (metres)."""
    if not raw:
        return pd.Series(dtype=float, name='observed')
    try:
        contents = gzip.decompress(raw).decode('ascii', errors='replace')
    except (gzip.BadGzipFile, OSError):
        contents = raw.decode('ascii', errors='replace')
    headers, records = None, []
    for ln in contents.splitlines():
        items = ln.strip().split()
        if not items:
            continue
        if items[0].lstrip('#').lower() in ('yy', 'yyyy', 'yr') and ('WVHT' in items):
            headers = [i.lstrip('#') for i in items]
            continue
        if headers is None or ln.lstrip().startswith('#'):
            continue
        if len(items) < len(headers):
            continue
        try:
            rec = dict(zip(headers, items))
            yr = int(rec.get('YY', rec.get('YYYY', rec.get('yr', year))))
            if yr < 100:
                # Annual file name disambiguates 2-digit year precisely.
                yr = (year // 100) * 100 + yr
                if abs(yr - year) > 50:
                    yr += 100 if yr < year else -100
            mon = int(rec['MM'])
            day = int(rec['DD'])
            hour = int(rec.get('hh', rec.get('HH', 0)))
            minute = int(rec.get('mm', 0))
            ts = pd.Timestamp(year=yr, month=mon, day=day,
                              hour=hour, minute=minute, tz='UTC')
            h = float(rec['WVHT'])
            if math.isfinite(h) and 0 <= h <= 30 and h not in (99.0, 999.0):
                records.append((ts, h))
        except (TypeError, ValueError, KeyError, OverflowError):
            pass
    if not records:
        return pd.Series(dtype=float, name='observed')
    z = pd.Series([p[1] for p in records],
                  index=pd.DatetimeIndex([p[0] for p in records]), name='observed')
    z = z[~z.index.duplicated(keep='first')].sort_index()
    # Historic NDBC measurements often occur at XX:50; assign to closest UTC hour.
    z.index = z.index.round('h')
    z = z.groupby(level=0).mean().sort_index()
    return z


def ndbc_year(sess, station_id, year, cache_dir):
    csv_path = cache_dir / 'ndbc_parsed' / f'{station_id}_{year}.csv'
    if csv_path.exists():
        return read_ts_csv(csv_path)
    url = NDBC_STD_URL.format(station_id, year)
    raw = cached_request_bytes(sess, url,
                               cache_dir/'ndbc_raw'/f'{station_id}h{year}.txt.gz')
    if raw is None:
        return pd.Series(dtype=float, name='observed')
    ts = parse_ndbc_stdmet(raw, year)
    write_ts_csv(ts, csv_path)
    return ts


def clean_timeseries(df, name, low, high):
    if df.empty:
        return pd.Series(dtype=float, name=name)
    idx = pd.to_datetime(df['time'], errors='coerce', utc=True)
    vals = pd.to_numeric(df['value'], errors='coerce')
    result = pd.Series(vals.to_numpy(dtype=float), index=idx, name=name)
    result = result.loc[result.index.notna()].sort_index()
    result = result[(result >= low) & (result <= high)]
    return result.groupby(level=0).mean()


def write_ts_csv(s, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    s.rename('value').to_frame().to_csv(path, index_label='time')


def read_ts_csv(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.Series(dtype=float, name='observed')
        return clean_timeseries(df, 'observed', -30, 35)
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return pd.Series(dtype=float, name='observed')


def download_obs(sess, station, start_year, end_year, cache_dir):
    parts = []
    for year in range(start_year, end_year+1):
        if station['type'] == 'WL':
            s = coops_year(sess, station['id'], year, cache_dir)
        else:
            s = ndbc_year(sess, station['id'], year, cache_dir)
        if not s.empty:
            parts.append(s)
    if not parts:
        return pd.Series(dtype=float, name='observed')
    result = pd.concat(parts).sort_index()
    return result.loc[~result.index.duplicated(keep='first')]


def open_cora_catalog():
    import intake
    import zarr
    version = int(zarr.__version__.split('.')[0])
    cat = 'CORA_intake_zarr3.yml' if version >= 3 else 'CORA_intake.yml'
    LOG.info('Opening CORA %s from NOAA public AWS bucket', cat)
    return intake.open_catalog(CORA_S3 + cat, storage_options={'anon': True})


class CoraExtractor:
    """Open each dataset once and build a spherical nearest-node tree once."""
    def __init__(self, catalog):
        self.cat = catalog
        self.datasets = {}
        self.trees = {}
        self.coords = {}

    def prepare(self, kind):
        if kind in self.datasets:
            return
        ds = self.cat[CORA_CATALOG[kind]].to_dask()
        if CORA_VARIABLE[kind] not in ds:
            raise KeyError(f'{kind} missing variable {CORA_VARIABLE[kind]}: {list(ds)}')
        # x,y are longitude and latitude in the native ADCIRC grid.
        lon = np.asarray(ds['x'].values, dtype=float).reshape(-1)
        lat = np.asarray(ds['y'].values, dtype=float).reshape(-1)
        valid = np.where(np.isfinite(lat) & np.isfinite(lon) &
                         (np.abs(lat) <= 90) & (np.abs(lon) <= 180))[0]
        if not len(valid):
            raise ValueError('CORA node coordinates are not geographic degrees')
        self.datasets[kind] = ds
        self.coords[kind] = (lon, lat, valid)
        self.trees[kind] = cKDTree(unit_sphere(lat[valid], lon[valid]))
        LOG.info('CORA %s ready: %s spatial nodes', kind, len(valid))

    def fetch(self, kind, sta, start_year, end_year, cache_dir):
        filename = f'{kind}_{sta["id"]}_{start_year}_{end_year}.csv'
        cached = cache_dir/'cora_parsed'/filename
        attr_path = cache_dir/'cora_parsed'/filename.replace('.csv', '.json')
        if cached.exists() and attr_path.exists():
            d = json.loads(attr_path.read_text())
            return read_ts_csv(cached).rename('cora'), d

        self.prepare(kind)
        lon, lat, valid = self.coords[kind]
        dist, loc = self.trees[kind].query(unit_sphere([sta['lat']], [sta['lon']])[0], k=1)
        node = int(valid[int(loc)])
        distance = float(haversine_km(sta['lat'], sta['lon'], lat[node], lon[node]))
        if distance > MAX_NODE_DISTANCE_KM:
            raise ValueError(f'CORA {kind} node is {distance:.1f} km from {sta["id"]}; '
                             f'limit is {MAX_NODE_DISTANCE_KM} km')
        ds = self.datasets[kind]
        da = ds[CORA_VARIABLE[kind]].isel(node=node).sel(
            time=slice(f'{start_year}-01-01', f'{end_year}-12-31T23:59:59'))
        LOG.info('Fetching CORA %s %s: node=%d, offset=%.2f km, years=%d-%d',
                 kind, sta['id'], node, distance, start_year, end_year)
        loaded = da.compute()  # ONLY selected node x available period, not all nodes
        t = pd.to_datetime(loaded.time.values, utc=True, errors='coerce')
        v = np.asarray(loaded.values).reshape(-1).astype(float)
        units = str(da.attrs.get('units', 'm')).lower().strip()
        if units in ('mm', 'millimeters', 'millimetres'):
            v = v / 1000
        elif units not in ('m', 'meters', 'metres', 'meter', 'metre', ''):
            LOG.warning('Unexpected CORA %s unit %r; inspect NOAA metadata', kind, units)
        arr = pd.DataFrame({'time':t, 'value':v})
        ts = clean_timeseries(arr, 'cora', -30 if kind == 'WL' else 0,
                              30 if kind == 'WL' else 35)
        attr = dict(node=node, station_id=sta['id'], station_lat=sta['lat'],
                    station_lon=sta['lon'], cora_node_lat=float(lat[node]),
                    cora_node_lon=float(lon[node]), node_offset_km=round(distance, 3),
                    units='m', first=str(ts.index.min()) if len(ts) else None,
                    last=str(ts.index.max()) if len(ts) else None)
        write_ts_csv(ts, cached)
        attr_path.write_text(json.dumps(attr, indent=2))
        return ts, attr


def aligned(obs, model, start_year, end_year):
    """Strict UTC hourly intersection. Never fill gaps or time-shift for skill."""
    if obs.empty or model.empty:
        return pd.DataFrame(columns=['observed', 'cora'])
    ob = obs.rename('observed').dropna()
    si = model.rename('cora').dropna()
    # CO-OPS hourly data is hourly; NDBC is rounded in its parser.
    paired = pd.concat([ob, si], axis=1, join='inner').dropna().sort_index()
    paired = paired.loc[(paired.index.year >= start_year) &
                        (paired.index.year <= end_year)]
    return paired[~paired.index.duplicated(keep='first')]


def metrics(pair, start_year, end_year):
    coverage_hours = int((pd.Timestamp(f'{end_year+1}-01-01', tz='UTC') -
                          pd.Timestamp(f'{start_year}-01-01', tz='UTC')) / pd.Timedelta(hours=1))
    result = dict(n_pairs=len(pair), total_possible_hours=coverage_hours,
                  coverage_pct=round(100*len(pair)/coverage_hours, 2),
                  overlap_first=pair.index.min().isoformat() if len(pair) else '',
                  overlap_last=pair.index.max().isoformat() if len(pair) else '',
                  pearson_r2=np.nan, nse=np.nan, rmse_m=np.nan,
                  bias_m=np.nan, obs_mean_m=np.nan, cora_mean_m=np.nan)
    if len(pair) < 3:
        return result
    o, p = pair['observed'].values.astype(float), pair['cora'].values.astype(float)
    residual = p-o
    denom = np.sum((o - o.mean())**2)
    if denom > 0:
        result['nse'] = 1 - np.sum(residual**2) / denom
        if np.std(p) > 0:
            result['pearson_r2'] = float(np.corrcoef(o, p)[0,1]**2)
    result.update(rmse_m=float(np.sqrt(np.mean(residual**2))),
                  bias_m=float(np.mean(residual)),
                  obs_mean_m=float(np.mean(o)), cora_mean_m=float(np.mean(p)))
    return result


def annual_maxima(pair, start_year, end_year):
    rows = []
    for y in range(start_year, end_year+1):
        chunk = pair.loc[pair.index.year == y]
        possible = int((pd.Timestamp(f'{y+1}-01-01', tz='UTC') -
                        pd.Timestamp(f'{y}-01-01', tz='UTC'))/pd.Timedelta(hours=1))
        pct = len(chunk) / possible
        rows.append(dict(year=y, n_pair_hours=len(chunk),
                         coverage_fraction=round(pct, 4),
                         usable=bool(pct >= MIN_YEAR_COVERAGE),
                         obs_max_m=float(chunk['observed'].max()) if len(chunk) else np.nan,
                         cora_max_m=float(chunk['cora'].max()) if len(chunk) else np.nan,
                         max_bias_m=(float(chunk['cora'].max()-chunk['observed'].max())
                                     if len(chunk) else np.nan)))
    return pd.DataFrame(rows)


def observed_events(pair):
    """Observed independent high events and CORA max within +/-6 hours.

    Uses a fixed 72-hour gap between observed peaks; no event time shifting
    is used in hourly skill. This is an event-amplitude diagnostic only.
    """
    cols = ['peak_time_utc','obs_peak_m','cora_event_peak_m','lag_hours','event_bias_m']
    if len(pair) < 100:
        return pd.DataFrame(columns=cols)
    hourly = pair['observed'].asfreq('h')
    if hourly.notna().sum() < 100:
        return pd.DataFrame(columns=cols)
    q = float(hourly.quantile(EVENT_THRESHOLD_QUANTILE))
    padded = hourly.fillna(-1e20).to_numpy()
    peaks, _ = find_peaks(padded, height=q, distance=EVENT_DECLUSTER_HOURS)
    result = []
    for pos in peaks:
        event_t = hourly.index[pos]
        window = pair.loc[(pair.index >= event_t-pd.Timedelta(hours=EVENT_MATCH_WINDOW_HOURS)) &
                          (pair.index <= event_t+pd.Timedelta(hours=EVENT_MATCH_WINDOW_HOURS)), 'cora']
        if window.empty:
            continue
        pidx = window.idxmax()
        obs_peak = float(hourly.iloc[pos]); cora_peak = float(window.max())
        result.append((event_t, obs_peak, cora_peak,
                       float((pidx-event_t)/pd.Timedelta(hours=1)), cora_peak-obs_peak))
    return pd.DataFrame(result, columns=cols).sort_values('obs_peak_m', ascending=False)


def plotting_theme():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'figure.facecolor': 'white',
                         'savefig.facecolor': 'white', 'grid.alpha': 0.22})


def draw_ts(ax, pair, kind):
    if pair.empty:
        ax.text(.5,.5,'No paired observations',ha='center',va='center',transform=ax.transAxes)
        return
    # Daily means keep the full multi-decade time range visible. Validation
    # statistics and exported values retain their original hourly pairs.
    daily = pair.resample('1D').mean()
    ax.plot(daily.index, daily['observed'], color=OBS_COLOR, lw=.60, label='NOAA/NDBC obs')
    ax.plot(daily.index, daily['cora'], color=MODEL_COLOR, lw=.55, alpha=.75,
            label='CORA', zorder=2)
    ax.set_ylabel(LABEL[kind]); ax.set_xlabel('Year (daily means)')
    ax.set_title('A. Complete overlapping record')
    ax.grid(True); ax.legend(loc='upper right', ncol=2, fontsize=8)


def draw_events(ax, events, kind):
    if events.empty:
        ax.text(.5,.5,'Insufficient paired observations for events',
                ha='center',va='center',transform=ax.transAxes)
        return
    sub = events.head(EVENTS_TO_PLOT).iloc[::-1]
    y = np.arange(len(sub))
    ax.barh(y-.18, sub['obs_peak_m'], height=.34, color=OBS_COLOR, label='Obs peak')
    ax.barh(y+.18, sub['cora_event_peak_m'], height=.34, color=MODEL_COLOR,
            label=f'CORA max ±{EVENT_MATCH_WINDOW_HOURS} h')
    ax.set_yticks(y, pd.to_datetime(sub['peak_time_utc']).dt.strftime('%Y-%m-%d'))
    ax.set_xlabel(LABEL[kind]); ax.set_title('B. Top independent observed events')
    ax.grid(axis='x'); ax.legend(fontsize=8, loc='lower right')


def draw_scatter(ax, pair, kind, skill):
    if len(pair) < 3:
        ax.text(.5,.5,'No paired observations',ha='center',va='center',transform=ax.transAxes)
        return
    x, y = pair['observed'].to_numpy(), pair['cora'].to_numpy()
    vmin = min(np.min(x), np.min(y)); vmax = max(np.max(x), np.max(y))
    if abs(vmax-vmin) < 1e-5: vmax = vmin + .1
    h = ax.hexbin(x, y, gridsize=60, mincnt=1, cmap='viridis', norm=LogNorm(),
                  extent=(vmin, vmax, vmin, vmax))
    ax.plot([vmin,vmax],[vmin,vmax], 'k--', lw=1, label='1:1')
    ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Observed (m)'); ax.set_ylabel('CORA (m)')
    ax.set_title('C. Hourly observed vs CORA')
    ax.text(.03,.97, f'N = {len(pair):,}\nR² = {skill["pearson_r2"]:.3f}\n'
            f'NSE = {skill["nse"]:.3f}\nRMSE = {skill["rmse_m"]:.3f} m\n'
            f'Bias = {skill["bias_m"]:+.3f} m', transform=ax.transAxes,
            ha='left',va='top',fontsize=8,
            bbox=dict(boxstyle='round',facecolor='white',alpha=.88,edgecolor='gray'))


def draw_annual(ax, annual, kind):
    valid = annual.loc[annual['usable']].copy()
    if valid.empty:
        ax.text(.5,.5,f'No year with >= {MIN_YEAR_COVERAGE:.0%} paired coverage',
                ha='center',va='center',transform=ax.transAxes)
        return
    ax.plot(valid['year'],valid['obs_max_m'],marker='o',ms=3,color=OBS_COLOR,lw=1,
            label='Observed annual maximum')
    ax.plot(valid['year'],valid['cora_max_m'],marker='s',ms=3,color=MODEL_COLOR,lw=1,
            label='CORA annual maximum')
    ax.set_xlabel('Year'); ax.set_ylabel(LABEL[kind])
    ax.set_title(f'D. Annual maxima (paired coverage ≥ {MIN_YEAR_COVERAGE:.0%})')
    ax.grid(True); ax.legend(fontsize=8)


def plot_station(pair, annual, events, kind, skill, station, outdir):
    if pair.empty:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2,2,figsize=(15,10.5))
    draw_ts(axes[0,0],pair,kind)
    draw_events(axes[0,1],events,kind)
    draw_scatter(axes[1,0],pair,kind,skill)
    draw_annual(axes[1,1],annual,kind)
    fig.suptitle(f'CORA-GEC validation | {station["id"]}: {station["name"]} | '
                 f'{"Water level" if kind=="WL" else "Hs"}',
                 fontsize=15, weight='bold')
    fig.text(.5,.014, 'CORA sampled at nearest native-grid node to observation station | '
             'Hourly paired skill | UTC | CORA GEC 1979–2022', ha='center', fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.94],h_pad=3,w_pad=2)
    name=f'{kind}_{station["id"]}'
    saved=[outdir/f'{name}_overview.png']
    fig.savefig(saved[-1],dpi=180)
    # Also save the 4 requested charts separately from the same axes.
    fig.canvas.draw()
    rend=fig.canvas.get_renderer()
    for tag,ax in zip(('timeseries','event_maxima','scatter','annual_maxima'), axes.flat):
        bb=ax.get_tightbbox(rend).transformed(fig.dpi_scale_trans.inverted())
        bb=bb.expanded(1.12,1.13)
        fp=outdir/f'{name}_{tag}.png'
        fig.savefig(fp,dpi=180,bbox_inches=bb)
        saved.append(fp)
    plt.close(fig)
    return saved


def plot_summary(site_summary, outpath):
    plotting_theme()
    fig, axs=plt.subplots(2,2,figsize=(16,12))
    labels=[f'{r.site_id} {r.site}' for r in site_summary.itertuples()]
    y=np.arange(len(labels)); h=.34
    for col, kind in enumerate(('WL','HS')):
        r2=site_summary[f'{kind}_r2'].to_numpy(dtype=float)
        nse=site_summary[f'{kind}_nse'].to_numpy(dtype=float)
        rmse=site_summary[f'{kind}_rmse_m'].to_numpy(dtype=float)
        bias=site_summary[f'{kind}_bias_m'].to_numpy(dtype=float)
        ax=axs[0,col]
        ax.barh(y-h/2,r2,height=h,label='Pearson r²',color='#246591')
        ax.barh(y+h/2,nse,height=h,label='NSE',color='#e08a42')
        ax.axvline(0,color='gray',lw=.7); ax.axvline(1,color='gray',lw=.7,ls=':')
        ax.set_yticks(y, labels if col==0 else [str(z) for z in site_summary.site_id])
        ax.invert_yaxis(); ax.set_title(f'{kind}: correlation and efficiency')
        ax.set_xlabel('Dimensionless score'); ax.legend(fontsize=8)
        ax.grid(axis='x'); ax=axs[1,col]
        ax.barh(y-h/2,rmse,height=h,label='RMSE (m)',color='#246591')
        ax.barh(y+h/2,bias,height=h,label='Bias (m)',color='#e08a42')
        ax.axvline(0,color='gray',lw=.7)
        ax.set_yticks(y, labels if col==0 else [str(z) for z in site_summary.site_id])
        ax.invert_yaxis();ax.set_xlabel('Metres'); ax.set_title(f'{kind}: error magnitude')
        ax.grid(axis='x');ax.legend(fontsize=8)
    fig.suptitle('North Carolina CORA validation | all 10 study sites',
                 fontsize=16, weight='bold')
    fig.text(.5,.012,'Each site uses nearest station among the supplied candidates. '
             'Shared stations intentionally have identical skill. Bias = CORA − observation.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95],w_pad=3)
    outpath.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outpath,dpi=200);plt.close(fig)


def plot_coverage(station_stats, outpath):
    fig,ax=plt.subplots(figsize=(13,max(5,len(station_stats)*.4+2)))
    labels=[f'{r.kind} {r.station_id}' for r in station_stats.itertuples()]
    cov=station_stats.coverage_pct.fillna(0).to_numpy()
    y=np.arange(len(labels))
    ax.barh(y,cov,color=['#246591' if x=='WL' else '#e08a42' for x in station_stats.kind])
    ax.set_yticks(y,labels);ax.invert_yaxis()
    ax.set_xlabel('Paired-hour coverage of complete requested CORA window (%)')
    ax.set_title('Observation availability and CORA overlap (unique stations)',weight='bold')
    for yi,v in enumerate(cov):ax.text(v+.2,yi,f'{v:.1f}%',va='center',fontsize=8)
    ax.set_xlim(0,max(105,float(np.nanmax(cov))+10))
    ax.grid(axis='x');fig.tight_layout()
    fig.savefig(outpath,dpi=180);plt.close(fig)


def make_map(assign, stations, outdir):
    try:
        import folium
    except ImportError:
        LOG.warning('folium not installed; skipping HTML station assignment map')
        return
    m=folium.Map(location=[35.45,-76.55],zoom_start=7,tiles='CartoDB positron')
    for r in assign.itertuples(index=False):
        txt=(f'{r.site}<br>Nearest tide gauge: {r.WL_station_id} '
             f'({r.WL_distance_km:.1f} km)<br>Nearest wave buoy: '
             f'{r.HS_station_id} ({r.HS_distance_km:.1f} km)')
        folium.CircleMarker([r.site_lat,r.site_lon],radius=5,color='#245f9b',
            fill=True,fill_opacity=.9,popup=folium.Popup(txt,max_width=350),
            tooltip=f'{r.site_id} {r.site}').add_to(m)
        for kind in ('WL','HS'):
            p=stations[getattr(r,f'{kind}_station_id')]
            folium.PolyLine([[r.site_lat,r.site_lon],[p['lat'],p['lon']]],
                color='#2e74a4' if kind=='WL' else '#d2773d',
                weight=1,opacity=.25).add_to(m)
    for p in stations.values():
        color='#114b7a' if p['type']=='WL' else '#dc7338'
        folium.CircleMarker([p['lat'],p['lon']],radius=7,color=color,
            fill=True,fill_color=color,fill_opacity=.8,
            popup=f'{p["type"]}: {p["name"]} ({p["id"]})',
            tooltip=p['id']).add_to(m)
    m.save(str(outdir/'nc_sites_stations_map.html'))


def write_readme(outdir, args):
    txt=f'''CORA NC VALIDATION | METHODS AND INTERPRETATION

Requested CORA domain: GEC 1979-2022; analyzed years: {args.start_year}-{args.end_year}.
Water level: nearest native-grid CORA zeta to NOAA CO-OPS hourly_height gauge.
Waves: nearest native-grid CORA swan_HS to NDBC stdmet WVHT buoy.
All model spatial extraction locations are the OBSERVATION STATIONS, not the sites.

STATION MATCHING
Haversine geodesic proximity only among the 6 supplied CO-OPS stations and
10 supplied NDBC buoys. Nearest does not guarantee hydrodynamic suitability;
particularly for inland creeks, estuaries and inlets. Review map and distances.
Station coordinates are queried from NOAA and NDBC metadata sources.
Distance flags: tide gauges >30 km and wave buoys >50 km; these are
interpretative heuristics, NOT calibrated thresholds.

PREPROCESSING
CORA GEC: NOAA public S3 intake time-series catalog; native ADCIRC/SWAN grid;
coordinates from the dataset; nearest node sampled once per unique station.
CO-OPS: verified hourly_height, datum=MSL, units=metric, time_zone=GMT.
NDBC: historical annual standard meteorological WVHT, metres, record timestamp
rounded to nearest UTC hour; multiple reports per hour averaged. Inspect timestamps
if source records have systematic phase offsets or non-hourly sampling.
Only hours with BOTH observed and modeled values are used; no missing values
are filled and no datum/time bias correction is automatically applied.

STATISTICS
R2 is SQUARED PEARSON CORRELATION (not sklearn's r2_score; the latter equals NSE).
NSE = 1 - SUM((CORA-OBS)^2)/SUM((OBS-mean(OBS))^2).
RMSE = sqrt(mean((CORA-OBS)^2)) [m].
Bias = mean(CORA-OBS) [m]; positive indicates CORA overestimation.
Coverage percent = paired hours / all hours in the requested CORA period.

EVENTS AND ANNUAL MAXIMA
Observed event peaks: >={EVENT_THRESHOLD_QUANTILE:.0%} quantile, separated by >=
{EVENT_DECLUSTER_HOURS} hourly indices; CORA event maximum selected within +/-
{EVENT_MATCH_WINDOW_HOURS} hours of observed peak. This allows small peak timing errors.
Annual maxima are computed separately from observed and CORA series but restricted
to the same paired UTC hours. Calendar years are plotted only if >=
{MIN_YEAR_COVERAGE:.0%} of calendar-year hours are paired; lower coverage is shown
in annual maxima CSV with usable=False.
Time series graph displays DAILY MEANS for readability across decades; all
statistics and scatter plots use original HOURLY matched values.

RESEARCH QUALIFICATIONS
1. CORA water levels assimilate NOAA CO-OPS observations. Tide-gauge comparisons
   are therefore a consistency/skill check, NOT fully independent verification.
2. NOAA station MSL datum may have a different epoch or reference definition
   from CORA's MSL reference; inspect datum metadata before interpreting biases.
3. NDBC wave records vary by buoy and year; non-operational years and missing
   WVHT are not imputed. Reports may have subhourly timestamps.
4. The NDBC and CORA wave buoy/model-node depths and exposure can differ.
5. NOAA identifies CORA GEC wave flatlining in Nov 2001 and Aug 2004; these
   periods remain visible in raw exports and should be flagged or excluded
   in a secondary sensitivity study before scientific publication.
6. Where multiple sites share one station, identical validation metrics are
   repeated in site_summary.csv. Use station_skill.csv for independent counts.

OUTPUTS
site_station_mapping.csv         10 sites + closest observation stations; distance flags
all_candidate_distances.csv      Full geographic distance ranking
station_skill.csv                All 16 supplied station metrics and coverage (default)
site_summary.csv                 Metrics mapped back to 10 sites
station_node_locations.csv       CORA node location and offset for each station
annual_maxima/                  Annual observed/CORA maximum pairs
observed_events/                Observed event/CORA model-window maxima
paired_hourly/                  Complete paired hourly validation record
figures/stations/               Each unique station: 4 figures + combined
figures/sites/                  Each of 10 sites: 4+4 figures + combined
figures/overall_dashboard.png    Summary statistics, site level
figures/coverage.png             Paired-hour availability, station level
nc_sites_stations_map.html       Interactive assignment map

DATA SOURCES
https://noaa-nos-cora-pds.s3.amazonaws.com/README
https://github.com/NOAA-CO-OPS/CORA-Coastal-Ocean-Reanalysis-CORA
https://api.tidesandcurrents.noaa.gov/api/prod/
https://www.ndbc.noaa.gov/data/historical/stdmet/
https://www.ndbc.noaa.gov/data/stations/station_table.txt
'''
    (outdir/'METHODS_AND_README.txt').write_text(txt,encoding='utf-8')


def run(args):
    plotting_theme()
    args.start_year=max(GEC_FIRST,args.start_year)
    args.end_year=min(GEC_LAST,args.end_year)
    if args.start_year>args.end_year:
        raise ValueError('Requested date range does not intersect GEC 1979–2022')
    outdir=Path(args.output).resolve();outdir.mkdir(parents=True,exist_ok=True)
    cache=outdir/'cache';cache.mkdir(exist_ok=True)
    sess=session_with_retries()
    stations={}
    for code in COOPS:
        stations[code]=get_coops_metadata(sess,code,cache)
    stations.update(get_ndbc_metadata(sess,cache))
    assign, rank=assign_stations(SITES,stations)
    assign.to_csv(outdir/'site_station_mapping.csv',index=False)
    rank.to_csv(outdir/'all_candidate_distances.csv',index=False)
    make_map(assign,stations,outdir)
    write_readme(outdir,args)
    print('\nNEAREST STATIONS (selected candidates):\n')
    print(assign[['site_id','site','WL_station_id','WL_distance_km',
                  'HS_station_id','HS_distance_km']].to_string(index=False))
    if args.metadata_only:
        LOG.info('Metadata-only mode: no CORA or observational time series downloaded.')
        return

    catalog=open_cora_catalog()
    extractor=CoraExtractor(catalog)
    stats=[]; nodes={}; observed_data={}; pair_data={}; annual_data={}; event_data={}
    # By default validate ALL supplied observations (6 tide + 10 wave).
    # The nearest observation for each site is selected independently.
    if args.matched_only:
        chosen={(r[f'{kind}_station_id'],kind) for _,r in assign.iterrows()
                for kind in ('WL','HS')}
    else:
        chosen={(station['id'],station['type']) for station in stations.values()}
    for code,kind in sorted(chosen,key=lambda t:(t[1],t[0])):
        sta=stations[code]
        LOG.info('Processing %s %s (%s)',kind,code,sta['name'])
        try:
            obs=download_obs(sess,sta,args.start_year,args.end_year,cache)
            observed_data[(code,kind)]=obs
            model,node=extractor.fetch(kind,sta,args.start_year,args.end_year,cache)
            nodes[(code,kind)]=dict(kind=kind,**node)
            pair=aligned(obs,model,args.start_year,args.end_year)
            pair_data[(code,kind)]=pair
            skill=metrics(pair,args.start_year,args.end_year)
            stats.append(dict(station_id=code,kind=kind,station_name=sta['name'],
                              station_lat=sta['lat'],station_lon=sta['lon'],
                              cora_node=node['node'],node_offset_km=node['node_offset_km'],
                              **skill))
            ann=annual_maxima(pair,args.start_year,args.end_year)
            ev=observed_events(pair)
            annual_data[(code,kind)]=ann;event_data[(code,kind)]=ev
            (outdir/'paired_hourly').mkdir(exist_ok=True)
            pair.to_csv(outdir/'paired_hourly'/f'{kind}_{code}.csv',index_label='time_utc')
            (outdir/'annual_maxima').mkdir(exist_ok=True)
            ann.to_csv(outdir/'annual_maxima'/f'{kind}_{code}.csv',index=False)
            (outdir/'observed_events').mkdir(exist_ok=True)
            ev.to_csv(outdir/'observed_events'/f'{kind}_{code}.csv',index=False)
            if not pair.empty:
                plot_station(pair,ann,ev,kind,skill,sta,outdir/'figures'/'stations')
            else:
                LOG.warning('NO HOURLY OVERLAP for %s %s; no validation plots',kind,code)
        except Exception as exc:
            LOG.exception('Station %s %s failed, continuing with others: %s',kind,code,exc)
            stats.append(dict(station_id=code,kind=kind,station_name=sta['name'],
                              station_lat=sta['lat'],station_lon=sta['lon'],
                              status=f'ERROR: {exc}',n_pairs=0,coverage_pct=0))
    if not stats:
        LOG.error('Nothing was processed')
        return
    station_stats=pd.DataFrame(stats)
    station_stats.to_csv(outdir/'station_skill.csv',index=False)
    pd.DataFrame(nodes.values()).to_csv(outdir/'station_node_locations.csv',index=False)

    site_results=[]
    (outdir/'figures'/'sites').mkdir(parents=True,exist_ok=True)
    for _,r in assign.iterrows():
        row=dict(site_id=r['site_id'],site=r['site'])
        for kind in ('WL','HS'):
            code=r[f'{kind}_station_id']
            record=station_stats.loc[(station_stats.station_id==code)&
                                     (station_stats.kind==kind)]
            row[f'{kind}_station']=code
            row[f'{kind}_distance_km']=r[f'{kind}_distance_km']
            for orig,dest in [('pearson_r2','r2'),('nse','nse'),('rmse_m','rmse_m'),
                              ('bias_m','bias_m'),('n_pairs','n_pairs'),
                              ('coverage_pct','coverage_pct')]:
                row[f'{kind}_{dest}']=record.iloc[0].get(orig,np.nan) if len(record) else np.nan
            source=outdir/'figures'/'stations'
            dst=outdir/'figures'/'sites'/f'{r["site_id"]}_{slug(r["site"])}'
            dst.mkdir(exist_ok=True)
            for file in source.glob(f'{kind}_{code}_*.png'):
                shutil.copy2(file,dst/file.name)
        site_results.append(row)
    site_summary=pd.DataFrame(site_results)
    site_summary.to_csv(outdir/'site_summary.csv',index=False)
    plot_summary(site_summary,outdir/'figures'/'overall_dashboard.png')
    plot_coverage(station_stats,outdir/'figures'/'coverage.png')
    LOG.info('Finished. See %s',outdir)
    print('\nSITE SKILL SUMMARY\n')
    print(site_summary[['site_id','site','WL_station','WL_r2','WL_nse',
        'WL_rmse_m','HS_station','HS_r2','HS_nse','HS_rmse_m']].to_string(index=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start-year',type=int,default=GEC_FIRST)
    parser.add_argument('--end-year',type=int,default=GEC_LAST)
    parser.add_argument('--output',default='CORA_NC_validation_results')
    parser.add_argument('--metadata-only',action='store_true',
                        help='Only retrieve station metadata, nearest mappings and map')
    parser.add_argument('--matched-only',action='store_true',
                        help='Validate only unique stations matched to sites (faster); default validates all 16')
    a=parser.parse_args()
    run(a)


if __name__=='__main__':
    main()
