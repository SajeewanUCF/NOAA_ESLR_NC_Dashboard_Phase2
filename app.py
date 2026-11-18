"""North Carolina CORA–NOAA/NDBC validation dashboard.

Launch locally: streamlit run app.py
Deploy with GitHub: repository entrypoint app.py; requirements.txt at root.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import nc_cora_validation as core
from app_helpers import (
    analysis_folder, annual_figure, build_site_summary, coverage_figure,
    create_results_zip, event_figure, fallback_stations, import_precomputed_zip,
    load_sites, performance_figure, refresh_metadata, scatter_figure,
    station_skill, stored_result, timeseries_figure, validate_station,
)

st.set_page_config(page_title='NC CORA Validation | NOAA + NDBC',
                   page_icon='🌊', layout='wide', initial_sidebar_state='expanded')

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('CORA_DATA_DIR', str(APP_ROOT/'.local_data'))).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

st.markdown('''<style>
.block-container {padding-top: 1.45rem; max-width: 1500px;}
h1 {font-size: 2.0rem !important; letter-spacing: -.025em;}
[data-testid="stMetricValue"]{font-size: 1.55rem;}
small, .stCaption{color:#677889;}
</style>''', unsafe_allow_html=True)


def pretty_value(x, decimals=3, suffix=''):
    try:
        return f'{float(x):.{decimals}f}{suffix}' if pd.notna(x) else '—'
    except (ValueError, TypeError):
        return '—'


def fmt_station(s: dict):
    return f"{s['id']} — {s['name']}"


def results_plot_panel(run_dir: Path, kind: str, code: str):
    stored = stored_result(run_dir, kind, code)
    if stored is None:
        st.info('No saved paired-hour series for that station and analysis period. Run its validation first.')
        return
    pair, annual, events = stored
    if pair.empty:
        st.warning('The selected station has no overlapping valid hourly observations and CORA values.')
        return
    skills = station_skill(run_dir)
    record = skills[(skills.kind==kind)&(skills.station_id.astype(str)==str(code))]
    if not record.empty:
        stats=record.iloc[0]
        a,b,c,d,e=st.columns(5)
        a.metric('R² (Pearson)',pretty_value(stats.get('pearson_r2')))
        b.metric('NSE',pretty_value(stats.get('nse')))
        c.metric('RMSE',pretty_value(stats.get('rmse_m'),suffix=' m'))
        d.metric('Bias',pretty_value(stats.get('bias_m'),suffix=' m'))
        e.metric('Paired coverage',pretty_value(stats.get('coverage_pct'),1,'%'))
        st.caption(f"{int(stats.get('n_pairs',0)):,} strictly matched UTC hours | "
                   f"CORA native-grid node {stats.get('cora_node')} | "
                   f"node-to-observation distance {pretty_value(stats.get('node_offset_km'),2,' km')} | "
                   f"overlap: {stats.get('overlap_first','')} to {stats.get('overlap_last','')}")
    resolution=st.radio('Time-series display', ['Hourly','Daily','Monthly'],
                        index=1, horizontal=True, key=f'resolution_{kind}_{code}')
    f1=timeseries_figure(pair,kind,resolution)
    f2=event_figure(events,kind)
    f3=scatter_figure(pair,kind)
    f4=annual_figure(annual,kind)
    st.plotly_chart(f1,use_container_width=True, key=f'ts_{kind}_{code}')
    st.plotly_chart(f2,use_container_width=True, key=f'event_{kind}_{code}')
    st.plotly_chart(f3,use_container_width=True, key=f'scatter_{kind}_{code}')
    st.plotly_chart(f4,use_container_width=True, key=f'annual_{kind}_{code}')
    if kind=='HS' and 2001<=end_year and start_year<=2004 and exclude_problem_wave:
        st.caption('November 2001 and August 2004 were excluded from wave comparisons; annual maxima for 2001 and 2004 are marked unusable.')
    with st.expander('Inspect supporting tables'):
        st.markdown('**Annual maxima and coverage flags**')
        st.dataframe(annual,use_container_width=True,hide_index=True)
        st.markdown('**Largest observed events**')
        st.dataframe(events.head(50),use_container_width=True,hide_index=True)
    files=create_results_zip(run_dir,kind,code,{
        'time_series':f1,'event_maxima':f2,
        'observed_vs_cora':f3,'annual_maxima':f4})
    st.download_button('Download validation ZIP (CSV + interactive HTML charts)',
                       data=files,file_name=f'CORA_{kind}_{code}_{start_year}_{end_year}.zip',
                       mime='application/zip',key=f'zip_{kind}_{code}')


# ---------- Sidebar / user-defined study setup ----------
st.title('North Carolina CORA Validation')
st.caption('NOAA CORA-GEC model versus CO-OPS water levels and NDBC significant wave heights | '
           '10 North Carolina study sites | CORA 1979–2022')

with st.sidebar:
    st.header('Analysis settings')
    custom=st.file_uploader('Optional: upload site coordinates CSV',type=['csv'],
                            help='Columns: site_id, site, site_lat, site_lon')
    try:
        sites=load_sites(APP_ROOT/'data'/'study_sites.csv',custom.getvalue() if custom else None)
    except (ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f'Invalid site CSV: {exc}')
        st.stop()

    if 'station_metadata' not in st.session_state:
        st.session_state.station_metadata=fallback_stations()
        st.session_state.metadata_origin='Verified coordinates included in repository'
    stations=st.session_state.station_metadata
    assignments, candidates=core.assign_stations(sites,stations)
    site_id=st.selectbox('Study site',options=[s[0] for s in sites],
                         format_func=lambda v:next(f'{s[0]} — {s[1]}' for s in sites if s[0]==v))
    site_row=assignments.loc[assignments.site_id==site_id].iloc[0]
    requested_kinds=st.multiselect('Observation type',
                                   options=['WL','HS'],default=['WL','HS'],
                                   format_func=lambda v:'Water level (CO-OPS)' if v=='WL' else 'Wave Hs (NDBC)')
    use_override=st.checkbox('Manually override closest observation station',value=False)
    station_for={}
    for kind in ('WL','HS'):
        nearest=site_row[f'{kind}_station_id']
        choices=[s for s in stations.values() if s['type']==kind]
        choices=sorted(choices,key=lambda s:float(core.haversine_km(float(site_row.site_lat),
                              float(site_row.site_lon),s['lat'],s['lon'])))
        if use_override:
            code=st.selectbox('Tide gauge' if kind=='WL' else 'Wave buoy',
                              options=[s['id'] for s in choices],
                              format_func=lambda v:fmt_station(stations[v]),
                              index=[s['id'] for s in choices].index(nearest))
        else:
            code=nearest
        station_for[kind]=code
        distance=core.haversine_km(float(site_row.site_lat),float(site_row.site_lon),
                                   stations[code]['lat'],stations[code]['lon'])
        st.caption(f'{kind}: {code} · {distance:.1f} km from site')

    start_year,end_year=st.select_slider('Analysis years (CORA GEC)',
                                          options=list(range(1979,2023)),value=(2020,2022))
    exclude_problem_wave=st.checkbox('Exclude known 2001/2004 wave artefacts',value=True,
                                     help='Removes Nov 2001 and Aug 2004 for Hs; marks annual maxima for those years unusable.')
    st.caption('For full CORA availability select 1979–2022. Large jobs can exceed shared hosting resources.')

run_dir=analysis_folder(DATA_ROOT,start_year,end_year,exclude_problem_wave)

mapping_tab, validation_tab, summary_tab, methods_tab=st.tabs([
    'Stations & map','Run validation','Overall dashboard','Methodology & sources'])

with mapping_tab:
    st.subheader('Nearest candidate stations')
    st.write('Station selection uses great-circle proximity among the six supplied CO-OPS gauges and ten supplied NDBC buoys. '
             '**CORA extraction uses the selected station coordinates, never study-site coordinates.**')
    show_map=assignments.copy()
    st.dataframe(show_map[['site_id','site','WL_station_id','WL_distance_km',
                           'WL_distance_flag','HS_station_id','HS_distance_km','HS_distance_flag']],
                 use_container_width=True,hide_index=True)
    col1,col2=st.columns([3,1])
    with col1:
        st.caption(f'Station coordinate source: {st.session_state.metadata_origin}. '
                   'Distance flags (>30 km gauges / >50 km buoys) are screening indicators, not representativeness tests.')
    with col2:
        if st.button('Refresh NOAA/NDBC metadata',use_container_width=True):
            with st.spinner('Retrieving station coordinates from NOAA metadata services…'):
                st.session_state.station_metadata=refresh_metadata(DATA_ROOT/'source_cache')
                st.session_state.metadata_origin='NOAA API metadata with published fallbacks'
            st.rerun()
    try:
        (DATA_ROOT/'maps').mkdir(parents=True, exist_ok=True)
        core.make_map(assignments,stations,DATA_ROOT/'maps')
        html=(DATA_ROOT/'maps'/'nc_sites_stations_map.html').read_text(encoding='utf-8')
        components.html(html,height=560,scrolling=False)
    except Exception as exc:
        st.warning(f'Could not display the interactive map: {exc}')
        locs=pd.DataFrame([{'lat':s[2],'lon':s[3]} for s in sites] +
                          [{'lat':q['lat'],'lon':q['lon']} for q in stations.values()])
        st.map(locs)
    with st.expander('All candidate station distances for the selected site'):
        subset=candidates[candidates.site_id==site_id].sort_values(['variable','rank'])
        st.dataframe(subset,use_container_width=True,hide_index=True)
    st.download_button('Download study-site / nearest-station assignments',
                       assignments.to_csv(index=False).encode('utf-8'),
                       file_name='nc_site_station_assignments.csv',mime='text/csv')

with validation_tab:
    st.subheader(f"Validate {site_row.site}")
    st.write('Selected observation stations: '+', '.join(f'{k} {station_for[k]}' for k in requested_kinds) +
             f'. Assessment period: {start_year}–{end_year}.')
    selected_wl_distance = float(core.haversine_km(float(site_row.site_lat),float(site_row.site_lon),
                                      stations[station_for['WL']]['lat'],stations[station_for['WL']]['lon']))
    selected_hs_distance = float(core.haversine_km(float(site_row.site_lat),float(site_row.site_lon),
                                      stations[station_for['HS']]['lat'],stations[station_for['HS']]['lon']))
    if 'WL' in requested_kinds and selected_wl_distance>30:
        st.warning('The geographically nearest water-level gauge is more than 30 km from the selected site.')
    if 'HS' in requested_kinds and selected_hs_distance>50:
        st.warning('The geographically nearest wave buoy is more than 50 km from the selected site; '
                   'this does not establish local wave representativeness.')
    st.info('CORA native-grid node selection occurs at each observation station. '
            'Only exact hourly UTC pairs are scored. Downloading a new period from NOAA/CORA can be lengthy.')
    if st.button('Run validation for selected station(s)',type='primary',
                 disabled=(len(requested_kinds)==0),use_container_width=True):
        with st.status('Running selected comparisons',expanded=True) as status:
            any_error=False
            for kind in requested_kinds:
                station=stations[station_for[kind]]
                st.write(f'Downloading {kind} observations and CORA at {station["id"]}…')
                try:
                    result=validate_station(station,start_year,end_year,
                                             DATA_ROOT,exclude_problem_wave)
                    st.write(f"{kind} {station['id']}: {result['status']} "
                             f"({result.get('n_pairs',0):,} paired hours)")
                except Exception as exc:
                    any_error=True
                    st.error(f"{kind} {station['id']} failed: {exc}")
            status.update(label='Station processing finished' if not any_error else 'Processing finished with some errors',
                          state='complete' if not any_error else 'error')
    skills=station_skill(run_dir)
    if not skills.empty:
        ready=skills[(skills.status=='OK') & (skills.n_pairs.fillna(0)>0)].copy()
        if not ready.empty:
            display_codes=ready.apply(lambda r:f'{r.kind} • {r.station_id} • {r.station_name}',axis=1).tolist()
            ix=st.selectbox('Visualize a completed station validation',range(len(display_codes)),
                            format_func=lambda i:display_codes[i])
            chosen=ready.iloc[ix]
            results_plot_panel(run_dir,str(chosen.kind),str(chosen.station_id))
        else:
            st.warning('No station with paired observation–CORA hours has been processed for this period.')
    else:
        st.caption('No completed validation results yet for the selected year range. Use the run button above.')

    with st.expander('Batch run (resource-intensive)'):
        st.write('For a full ten-site assessment the nearest-station option processes unique station pairs once, '
                 'then reuses those validation results for sites assigned to the same station. '
                 'All supplied stations processes all 6 tide gauges + 10 buoys.')
        batch_type=st.radio('Stations to process',['Unique nearest stations','All supplied observation stations'],
                             horizontal=True)
        approved=st.checkbox('I understand this may require substantial downloads and runtime',value=False)
        if st.button('Start batch validation',disabled=not approved):
            if batch_type=='Unique nearest stations':
                jobs={(str(r[f'{k}_station_id']),k) for _,r in assignments.iterrows() for k in ('WL','HS')}
            else:
                jobs={(q['id'],q['type']) for q in stations.values()}
            jobs=sorted(jobs,key=lambda v:(v[1],v[0]))
            progress=st.progress(0)
            for i,(code,kind) in enumerate(jobs):
                try:
                    with st.spinner(f'Processing {kind} {code} ({i+1}/{len(jobs)})…'):
                        result=validate_station(stations[code],start_year,end_year,DATA_ROOT,
                                                exclude_problem_wave)
                    st.write(f'{kind} {code}: {result["status"]}')
                except Exception as exc:
                    st.error(f'{kind} {code} failed: {exc}')
                progress.progress((i+1)/len(jobs))
            st.success('Batch attempt completed. Open Overall dashboard for available results.')

with summary_tab:
    st.subheader(f'Overall results • {start_year}–{end_year}')
    with st.expander('Import previous app or CLI results (without redownloading CORA)'):
        st.write('Upload an exported **station validation ZIP** from this app, or a ZIP containing '
                 'the original CLI output directory (with `METHODS_AND_README.txt` and `station_skill.csv`). '
                 'Year range and wave exclusion setting must match the selected sidebar options.')
        imported_zip=st.file_uploader('Previously generated result ZIP',type=['zip'],key='import_archive')
        if st.button('Import results into current dashboard',disabled=(imported_zip is None)):
            try:
                count=import_precomputed_zip(imported_zip.getvalue(),run_dir,start_year,end_year,
                                             exclude_problem_wave)
                st.success(f'Imported {count} station records into this analysis period.')
            except Exception as exc:
                st.error(f'Import rejected: {exc}')
    skills=station_skill(run_dir)
    if skills.empty:
        st.info('No downloaded validation results for this period. Use Run validation to populate the dashboard.')
    else:
        summary=build_site_summary(assignments,skills)
        done=skills[(skills.status=='OK')&(skills.n_pairs.fillna(0)>0)]
        a,b,c=st.columns(3)
        a.metric('Unique stations evaluated',f'{len(done)} / {len(stations)}')
        b.metric('Water-level gauges',str((done.kind=='WL').sum()))
        c.metric('Wave buoys',str((done.kind=='HS').sum()))
        st.caption('Ten-site rows may have identical statistics because several sites share one observation station. '
                   'Never treat the repeated rows as independent station validations.')
        st.markdown('**Site validation summary**')
        show=['site_id','site','WL_station_id','WL_distance_km','WL_r2','WL_nse',
              'WL_rmse_m','WL_bias_m','HS_station_id','HS_distance_km','HS_r2',
              'HS_nse','HS_rmse_m','HS_bias_m']
        st.dataframe(summary[show],use_container_width=True,hide_index=True)
        st.plotly_chart(performance_figure(summary,'WL'),use_container_width=True)
        st.plotly_chart(performance_figure(summary,'HS'),use_container_width=True)
        st.plotly_chart(coverage_figure(skills),use_container_width=True)
        with st.expander('Independent station-level results'):
            st.dataframe(skills,use_container_width=True,hide_index=True)
        d1,d2=st.columns(2)
        with d1:
            st.download_button('Download site summary CSV',summary.to_csv(index=False).encode(),
                               file_name=f'nc_cora_sites_{start_year}_{end_year}.csv',mime='text/csv')
        with d2:
            st.download_button('Download independent-station statistics CSV',skills.to_csv(index=False).encode(),
                               file_name=f'nc_cora_stations_{start_year}_{end_year}.csv',mime='text/csv')

with methods_tab:
    st.subheader('Methods and scientific interpretation')
    st.markdown('''
**Input and matching** — For each site, candidates are ranked by Haversine great-circle distance.
Only the six supplied CO-OPS gauges and ten NDBC buoy identifiers are eligible.
For each selected observation station, CORA is sampled at its closest native-grid node,
subject to the **25 km node-offset check**. This is not an interpolation to the site.

**Water-level reference** — CO-OPS `hourly_height`, `datum=MSL`, metric, GMT/UTC.
CORA GEC `fort.63` variable `zeta`, metres relative to model MSL. Verify reference epochs
before interpreting absolute offsets. CORA assimilates verified CO-OPS hourly gauge data,
so tide-gauge comparisons are not completely independent validation.

**Wave-height reference** — NDBC annual historical standard meteorological `WVHT` compared
with CORA GEC `swan_HS.63` `swan_HS`. NDBC subhourly reports are rounded to nearest
UTC hour. Compare bathymetry, depths and station exposures before generalizing skill.

**Quality control** — No gap filling, automatic datum adjustment, or lag correction for the
hourly scores. Analysis is limited to actual coincident hours within the chosen year window.
Optionally exclude the known CORA wave flatlining periods (November 2001, August 2004).
If excluded, the whole 2001 and 2004 annual maximum is marked unusable.

**Reported statistics** — R² is *squared Pearson correlation*, not coefficient of determination
computed as 1 − SSE/SST. The latter is the NSE used here. RMSE and bias are in metres;
bias is CORA − observed. Coverage is paired valid hours divided by all hours in the
requested CORA window (including missing observation years).

**Event comparison** — Observed peaks exceed the observed 98th-percentile threshold and are
separated by ≥72 hours. The CORA peak is taken within ±6 hours of the observed event peak.
This amplitude window is not used to shift the time series for hourly metrics.

**Annual maxima** — Observed and CORA annual maxima are computed from the *same paired-hour subset*.
Only calendar years with ≥75% valid paired hours are plotted, with flagged exceptions above.

**Cloud-hosting caveat** — The source archive is very large. The app loads selected node
records lazily, but retrieving a full 44-year record for many stations can be expensive in
time and memory. For production/research-grade full runs, use the bundled CLI on a workstation
or an appropriately sized VM and present the stored results in a Streamlit deployment.
''')
    st.markdown('**Official sources:** [CORA repository](https://github.com/NOAA-CO-OPS/CORA-Coastal-Ocean-Reanalysis-CORA) · '
                '[CORA data README](https://noaa-nos-cora-pds.s3.amazonaws.com/README) · '
                '[CO-OPS API](https://api.tidesandcurrents.noaa.gov/api/prod/) · '
                '[NDBC historic data](https://www.ndbc.noaa.gov/data/historical/stdmet/)')
    st.caption('Independent research dashboard; not an official NOAA application or NOAA-endorsed product.')
