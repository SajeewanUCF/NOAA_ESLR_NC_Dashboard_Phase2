"""Streamlit-facing orchestration around the reusable NC CORA validation engine.

No web downloads occur at import time.  Network requests only run after a user
explicitly requests NOAA station refresh or time-series validation.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nc_cora_validation as core

BLUE = '#20649B'
ORANGE = '#EC8642'
INK = '#10263A'
PAPER = '#FFFFFF'


def fallback_stations() -> dict[str, dict]:
    station_data = {
        code: dict(id=code, name=name, lat=core.COOPS_COORDS_FALLBACK[code][0],
                   lon=core.COOPS_COORDS_FALLBACK[code][1], type='WL')
        for code, name in core.COOPS.items()
    }
    station_data.update({
        code: dict(id=code, name=values[0], lat=values[1], lon=values[2], type='HS')
        for code, values in core.NDBC.items()
    })
    return station_data


def refresh_metadata(cache_path: Path) -> dict[str, dict]:
    sess = core.session_with_retries()
    stations = {
        code: core.get_coops_metadata(sess, code, cache_path)
        for code in core.COOPS
    }
    stations.update(core.get_ndbc_metadata(sess, cache_path))
    return stations


def load_sites(path: Path, csv_bytes: bytes | None = None) -> list[tuple[str, str, float, float]]:
    data = pd.read_csv(BytesIO(csv_bytes)) if csv_bytes is not None else pd.read_csv(path, dtype={'site_id':str})
    mandatory = {'site_id', 'site', 'site_lat', 'site_lon'}
    if not mandatory.issubset(data.columns):
        raise ValueError(f'Sites CSV requires columns {", ".join(sorted(mandatory))}.')
    if data.empty:
        raise ValueError('At least one study site is required.')
    data = data.copy()
    data['site_id'] = data['site_id'].astype(str).str.strip().str.zfill(2)
    data['site'] = data['site'].astype(str).str.strip()
    for c in ('site_lat','site_lon'):
        data[c] = pd.to_numeric(data[c], errors='raise')
    if data['site_id'].duplicated().any():
        raise ValueError('Each site_id must be unique.')
    if ((data.site_lat.abs() > 90) | (data.site_lon.abs() > 180)).any():
        raise ValueError('Site coordinates must be geographic latitude/longitude degrees.')
    return [(r.site_id, r.site, float(r.site_lat), float(r.site_lon)) for r in data.itertuples()]


def analysis_folder(base: Path, start: int, end: int, exclude_bad_wave: bool) -> Path:
    return base / 'runs' / f'{start}-{end}_wave_issue_excluded-{int(exclude_bad_wave)}'


def frame_or_empty(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def station_skill(run_folder: Path) -> pd.DataFrame:
    df = frame_or_empty(run_folder / 'station_skill.csv', dtype={'station_id':str})
    return df


def stored_result(run_folder: Path, kind: str, code: str):
    pair_file = run_folder / 'paired_hourly' / f'{kind}_{code}.csv'
    annual_file = run_folder / 'annual_maxima' / f'{kind}_{code}.csv'
    event_file = run_folder / 'observed_events' / f'{kind}_{code}.csv'
    if not pair_file.exists():
        return None
    pair = pd.read_csv(pair_file, parse_dates=['time_utc'], index_col='time_utc')
    pair.index = pd.to_datetime(pair.index, utc=True)
    annual = frame_or_empty(annual_file)
    events = frame_or_empty(event_file)
    if 'peak_time_utc' in events.columns:
        events['peak_time_utc'] = pd.to_datetime(events['peak_time_utc'], utc=True)
    return pair, annual, events


def _record_status(run_folder: Path, record: dict):
    run_folder.mkdir(parents=True, exist_ok=True)
    dest = run_folder / 'station_skill.csv'
    df = station_skill(run_folder)
    if not df.empty:
        df = df.loc[~((df.station_id.astype(str) == str(record['station_id'])) &
                      (df.kind == record['kind']))].copy()
    df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    df = df.sort_values(['kind','station_id']).reset_index(drop=True)
    df.to_csv(dest, index=False)


def _exclude_wave_artifacts(pair: pd.DataFrame) -> pd.DataFrame:
    if pair.empty:
        return pair
    affected = (((pair.index.year == 2001) & (pair.index.month == 11)) |
                ((pair.index.year == 2004) & (pair.index.month == 8)))
    return pair.loc[~affected].copy()


def validate_station(station: dict, start_year: int, end_year: int,
                     base: Path, exclude_bad_wave: bool = True) -> dict:
    """Download or retrieve point series and compute scientific comparisons.

    Station-specific observations are never substituted with study-site values.
    One station is processed at a time so cloud users can control runtime.
    """
    kind, code = station['type'], station['id']
    run_folder = analysis_folder(base, start_year, end_year, exclude_bad_wave)
    source_cache = base / 'source_cache'
    session = core.session_with_retries()
    obs = core.download_obs(session, station, start_year, end_year, source_cache)

    record = dict(station_id=str(code), station_name=station['name'], kind=kind,
                  station_lat=float(station['lat']), station_lon=float(station['lon']),
                  years=f'{start_year}-{end_year}',
                  exclude_problem_wave_months=bool(exclude_bad_wave))
    if obs.empty:
        record.update(status='NO_OBSERVATIONS', n_pairs=0, coverage_pct=0.0)
        _record_status(run_folder, record)
        return record

    extractor = core.CoraExtractor(core.open_cora_catalog())
    model, node_info = extractor.fetch(kind, station, start_year, end_year, source_cache)
    pair = core.aligned(obs, model, start_year, end_year)
    if kind == 'HS' and exclude_bad_wave:
        pair = _exclude_wave_artifacts(pair)
    skill = core.metrics(pair, start_year, end_year)
    annual = core.annual_maxima(pair, start_year, end_year)
    if kind == 'HS' and exclude_bad_wave:
        # Excluding an entire month can remove the true annual peak. Do not
        # interpret remaining maxima as complete annual maxima for these years.
        annual.loc[annual.year.isin([2001, 2004]), 'usable'] = False
    events = core.observed_events(pair)
    record.update(status='OK' if len(pair) else 'NO_PAIRED_HOURS',
                  cora_node=node_info['node'],
                  node_offset_km=node_info['node_offset_km'],
                  cora_node_lat=node_info['cora_node_lat'],
                  cora_node_lon=node_info['cora_node_lon'], **skill)

    for sub in ('paired_hourly','annual_maxima','observed_events','station_nodes'):
        (run_folder/sub).mkdir(parents=True, exist_ok=True)
    pair.to_csv(run_folder/'paired_hourly'/f'{kind}_{code}.csv', index_label='time_utc')
    annual.to_csv(run_folder/'annual_maxima'/f'{kind}_{code}.csv', index=False)
    events.to_csv(run_folder/'observed_events'/f'{kind}_{code}.csv', index=False)
    (run_folder/'station_nodes'/f'{kind}_{code}.json').write_text(json.dumps(node_info, indent=2))
    _record_status(run_folder, record)
    return record


def build_site_summary(assignments: pd.DataFrame, skills: pd.DataFrame) -> pd.DataFrame:
    summary = assignments[['site_id','site','WL_station_id','WL_distance_km',
                           'HS_station_id','HS_distance_km','WL_distance_flag',
                           'HS_distance_flag']].copy()
    for kind in ('WL','HS'):
        for k in ('pearson_r2','nse','rmse_m','bias_m','coverage_pct','n_pairs',
                  'node_offset_km','status'):
            col = f'{kind}_{"r2" if k == "pearson_r2" else k}'
            records = (skills.loc[skills.kind == kind, ['station_id', k]].copy()
                       if not skills.empty and k in skills.columns
                       else pd.DataFrame(columns=['station_id', k]))
            lookup = records.drop_duplicates('station_id').set_index('station_id')[k].to_dict()
            summary[col] = summary[f'{kind}_station_id'].astype(str).map(lookup)
    return summary


def _style(fig, title: str, y_title=''):
    fig.update_layout(title=dict(text=title, x=0.01, font=dict(size=17)),
                      font=dict(family='Arial', size=12, color=INK),
                      paper_bgcolor=PAPER, plot_bgcolor=PAPER,
                      margin=dict(l=55, r=24, t=65, b=60),
                      hovermode='closest', legend=dict(orientation='h', y=1.06, x=0))
    fig.update_xaxes(showgrid=True, gridcolor='#E8EEF3', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#E8EEF3', zeroline=False)
    if y_title:
        fig.update_yaxes(title_text=y_title)
    return fig


def timeseries_figure(pair: pd.DataFrame, kind: str, resolution: str = 'Daily'):
    if resolution == 'Hourly' and len(pair) > 90000:
        resolution = 'Daily'  # avoid sending huge browser payloads
    if resolution == 'Daily':
        view = pair.resample('D').mean().dropna()
    elif resolution == 'Monthly':
        view = pair.resample('MS').mean().dropna()
    else:
        view = pair
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=view.index, y=view['observed'], mode='lines',
                               name='Observed', line=dict(color=BLUE, width=1.3)))
    fig.add_trace(go.Scattergl(x=view.index, y=view['cora'], mode='lines',
                               name='CORA', line=dict(color=ORANGE, width=1.25)))
    fig.update_layout(hovermode='x unified')
    return _style(fig, f'Overlapping record • {resolution.lower()} means (metrics use paired hours)',
                  'Water level (m)' if kind=='WL' else 'Hs (m)')


def _identity_range(*values):
    z = np.concatenate([np.asarray(x, dtype=float).ravel() for x in values])
    z = z[np.isfinite(z)]
    if not len(z):
        return (-1,1)
    lo, hi = float(np.min(z)), float(np.max(z))
    pad = 0.05*max(hi-lo, 0.1)
    return lo-pad, hi+pad


def scatter_figure(pair: pd.DataFrame, kind: str, max_points: int = 18000):
    sampled = pair.sample(n=min(len(pair), max_points), random_state=1729) if len(pair)>max_points else pair
    lo, hi = _identity_range(pair['observed'], pair['cora'])
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=sampled['observed'], y=sampled['cora'], mode='markers',
                               marker=dict(size=3.5, color=BLUE, opacity=.24),
                               name='Hourly pairs', hovertemplate='Observed %{x:.3f} m<br>CORA %{y:.3f} m<extra></extra>'))
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode='lines', name='1:1',
                             line=dict(color=ORANGE,width=2,dash='dash')))
    fig.update_layout(xaxis_title='Observed (m)', yaxis_title='CORA (m)')
    fig.update_xaxes(range=[lo,hi]); fig.update_yaxes(range=[lo,hi],scaleanchor='x',scaleratio=1)
    return _style(fig, f'Observed versus CORA • {len(sampled):,} displayed of {len(pair):,} paired hours')


def event_figure(events: pd.DataFrame, kind: str, top_events: int = 20):
    fig = make_subplots(rows=1,cols=2,subplot_titles=('Largest observed events','Event peak comparison'))
    if not events.empty:
        ev=events.nlargest(top_events, 'obs_peak_m').sort_values('obs_peak_m',ascending=False).copy()
        labels = [str(pd.Timestamp(v).date()) for v in ev.peak_time_utc]
        fig.add_trace(go.Bar(x=labels,y=ev.obs_peak_m,name='Observed', marker_color=BLUE),1,1)
        fig.add_trace(go.Bar(x=labels,y=ev.cora_event_peak_m,name='CORA ±6 h', marker_color=ORANGE),1,1)
        lo,hi=_identity_range(events.obs_peak_m,events.cora_event_peak_m)
        fig.add_trace(go.Scatter(x=events.obs_peak_m,y=events.cora_event_peak_m, mode='markers',
                                 marker=dict(color=BLUE,size=7,opacity=.65),showlegend=False,
                                 text=events.peak_time_utc.astype(str),hovertemplate='%{text}<br>Observed %{x:.2f} m<br>CORA %{y:.2f} m<extra></extra>'),1,2)
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',
                                 line=dict(color=ORANGE,dash='dash'),showlegend=False),1,2)
    fig.update_layout(barmode='group',height=500)
    fig.update_xaxes(tickangle=55,row=1,col=1)
    fig.update_xaxes(title_text='Observed event maximum (m)',row=1,col=2)
    fig.update_yaxes(title_text='Peak (m)',row=1,col=1)
    fig.update_yaxes(title_text='CORA event maximum (m)',row=1,col=2)
    return _style(fig,'High-event peaks • observed peaks separated by ≥72 h; CORA maximum within ±6 h')


def annual_figure(annual: pd.DataFrame, kind: str):
    fig=make_subplots(rows=1,cols=2,subplot_titles=('Annual maxima by year','Observed versus CORA annual maxima'))
    if not annual.empty and 'usable' in annual:
        usable=annual.loc[annual['usable'].astype(str).str.lower().isin(['true','1'])].copy()
        if not usable.empty:
            fig.add_trace(go.Scatter(x=usable.year,y=usable.obs_max_m,mode='lines+markers',
                                     name='Observed annual maximum',line=dict(color=BLUE)),1,1)
            fig.add_trace(go.Scatter(x=usable.year,y=usable.cora_max_m,mode='lines+markers',
                                     name='CORA annual maximum',line=dict(color=ORANGE)),1,1)
            fig.add_trace(go.Scatter(x=usable.obs_max_m,y=usable.cora_max_m,mode='markers',
                                     marker=dict(color=BLUE,size=8),text=usable.year,
                                     hovertemplate='Year %{text}<br>Observed %{x:.3f} m<br>CORA %{y:.3f} m<extra></extra>',
                                     showlegend=False),1,2)
            lo,hi=_identity_range(usable.obs_max_m,usable.cora_max_m)
            fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',
                                     line=dict(color=ORANGE,dash='dash'),showlegend=False),1,2)
    fig.update_xaxes(title_text='Year',row=1,col=1)
    fig.update_xaxes(title_text='Observed annual max (m)',row=1,col=2)
    fig.update_yaxes(title_text='Annual peak (m)',row=1,col=1)
    fig.update_yaxes(title_text='CORA annual max (m)',row=1,col=2)
    fig.update_layout(height=520)
    return _style(fig,f'Annual maxima • years with ≥{core.MIN_YEAR_COVERAGE:.0%} paired-hour coverage')


def coverage_figure(skills: pd.DataFrame):
    s=skills.copy().sort_values(['kind','station_id'])
    if s.empty:
        return go.Figure()
    s['coverage_pct']=pd.to_numeric(s['coverage_pct'],errors='coerce').fillna(0)
    fig=go.Figure(go.Bar(y=(s.kind+' '+s.station_id),x=s.coverage_pct,orientation='h',
                         marker_color=[BLUE if x=='WL' else ORANGE for x in s.kind],
                         text=s.coverage_pct.map(lambda v:f'{v:.1f}%'),textposition='outside'))
    fig.update_layout(xaxis_title='Paired hours / all requested hours (%)',height=max(350, len(s)*35+130))
    fig.update_xaxes(range=[0,108])
    fig.update_yaxes(autorange='reversed')
    return _style(fig,'Observation and CORA overlapping-hour coverage')


def performance_figure(summary: pd.DataFrame, kind: str):
    t=summary.copy()
    x=t.site_id.astype(str).str.zfill(2) + ' ' + t.site.str.replace('Town of ','',regex=False)
    fig=make_subplots(rows=2,cols=2,subplot_titles=('Squared Pearson correlation','Nash–Sutcliffe efficiency',
                                                     'Root mean squared error (m)','Bias: CORA − observed (m)'))
    for r,c,attr,color in [(1,1,'r2',BLUE),(1,2,'nse',ORANGE),(2,1,'rmse_m',BLUE),(2,2,'bias_m',ORANGE)]:
        y=pd.to_numeric(t[f'{kind}_{attr}'],errors='coerce')
        fig.add_trace(go.Bar(x=x,y=y,marker_color=color,showlegend=False,
                             hovertemplate='%{x}<br>%{y:.3f}<extra></extra>'),row=r,col=c)
    fig.update_layout(height=830)
    fig.update_xaxes(tickangle=60)
    return _style(fig,f'{"Water level" if kind=="WL" else "Wave Hs"} • site assignments (shared stations repeat)')


def create_results_zip(run_folder: Path, kind: str, code: str,
                       figures: dict[str, go.Figure] | None = None) -> bytes:
    """Downloadable per-station evidence bundle, not a copy of remote CORA archive."""
    output=BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ('paired_hourly','annual_maxima','observed_events','station_nodes'):
            parent=run_folder/folder
            for path in parent.glob(f'{kind}_{code}.*') if parent.exists() else []:
                z.write(path,arcname=f'{folder}/{path.name}')
        skills=station_skill(run_folder)
        if not skills.empty:
            selected=skills[(skills.kind==kind)&(skills.station_id.astype(str)==str(code))]
            z.writestr('station_skill.csv',selected.to_csv(index=False))
        for name,fig in (figures or {}).items():
            z.writestr('charts/'+re.sub('[^a-z0-9_]+','_',name.lower())+'.html',
                       fig.to_html(include_plotlyjs='cdn',full_html=True))
    return output.getvalue()


def import_precomputed_zip(data: bytes, run_folder: Path, start_year: int,
                           end_year: int, exclude_bad_wave: bool) -> int:
    """Import app station ZIPs, or an archived output folder from the original CLI.

    The source's year range and known-wave-issue masking must match the selected
    dashboard options. Files are extracted only into allowlisted destinations.
    """
    with zipfile.ZipFile(BytesIO(data)) as archive:
        entries = {n: zi for n, zi in ((item.filename, item) for item in archive.infolist())
                   if not zi.is_dir()}
        if sum(zi.file_size for zi in entries.values()) > 450_000_000:
            raise ValueError('Archive uncompressed contents exceed 450 MB.')
        if any(zi.file_size > 250_000_000 for zi in entries.values()):
            raise ValueError('One archive member exceeds 250 MB.')

        def locate(suffix):
            found = [n for n in entries if n.replace('\\','/').split('/')[-1] == suffix]
            return found[0] if found else None

        skill_name = locate('station_skill.csv')
        if not skill_name:
            raise ValueError('ZIP must contain station_skill.csv from this project.')
        frame = pd.read_csv(BytesIO(archive.read(skill_name)), dtype={'station_id':str})
        if frame.empty or not {'station_id','kind'}.issubset(frame.columns):
            raise ValueError('station_skill.csv has no usable station identifiers.')
        readme_name=locate('METHODS_AND_README.txt')
        is_cli=bool(readme_name)
        if is_cli:
            info=archive.read(readme_name).decode('utf-8',errors='replace')
            m=re.search(r'analyzed years:\s*(\d{4})-(\d{4})',info,re.IGNORECASE)
            if not m or (int(m[1]),int(m[2]))!=(start_year,end_year):
                raise ValueError('Choose the same start/end years as the CLI report before importing.')
            if exclude_bad_wave:
                raise ValueError('CLI results did not exclude problem wave months. Uncheck the wave artefact exclusion option first.')
        else:
            if 'years' not in frame or 'exclude_problem_wave_months' not in frame:
                raise ValueError('Station ZIP lacks year range / wave exclusion metadata.')
            expected=f'{start_year}-{end_year}'
            if not (frame.years.astype(str)==expected).all():
                raise ValueError('ZIP data years differ from the selected dashboard year range.')
            mask_values=frame['exclude_problem_wave_months'].astype(str).str.lower().isin(('true','1'))
            if not (mask_values==bool(exclude_bad_wave)).all():
                raise ValueError('ZIP wave-exclusion mode differs from the dashboard checkbox.')

        imported=0
        for row in frame.itertuples(index=False):
            kind=str(row.kind)
            code=str(row.station_id)
            if kind not in ('WL','HS') or not code.isdigit() or len(code)>10:
                continue
            for section in ('paired_hourly','annual_maxima','observed_events','station_nodes'):
                target_name=f'{kind}_{code}.json' if section=='station_nodes' else f'{kind}_{code}.csv'
                candidates=[name for name in entries if name.replace('\\','/').lstrip('/').endswith(section+'/'+target_name)]
                if not candidates:
                    continue
                target=run_folder/section/target_name
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(archive.read(candidates[0]))
            rec=row._asdict()
            rec['status']=rec.get('status') if isinstance(rec.get('status'),str) and rec.get('status') else (
                'OK' if pd.to_numeric(rec.get('n_pairs',0),errors='coerce')>0 else 'NO_PAIRED_HOURS')
            rec.update(years=f'{start_year}-{end_year}',
                       exclude_problem_wave_months=bool(exclude_bad_wave))
            _record_status(run_folder,rec)
            imported+=1
        if imported==0:
            raise ValueError('ZIP did not contain valid WL/HS station results.')
        return imported
