"""Network-free regression checks for matching, scoring, charts and export."""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

import app_helpers as h
import nc_cora_validation as core

ROOT = Path(__file__).resolve().parents[1]


def test_coordinates_and_nearest():
    sites = h.load_sites(ROOT/'data'/'study_sites.csv')
    assignment, ranking = core.assign_stations(sites, h.fallback_stations())
    assert len(sites) == 10
    assert len(assignment)==10
    assert len(ranking)==10*16
    duck = assignment.loc[assignment.site_id=='09'].iloc[0]
    assert duck.WL_station_id=='8651370'
    assert duck.HS_station_id=='44056'
    assert duck.WL_distance_km<5


def test_metrics_and_annual():
    idx = pd.date_range('2020-01-01', '2020-12-31 23:00', freq='h',tz='UTC')
    obs = 0.5*np.sin(np.arange(len(idx))/12.4)
    pair=pd.DataFrame({'observed':obs,'cora':obs+0.1},index=idx)
    scores=core.metrics(pair,2020,2020)
    assert scores['n_pairs']==8784
    assert scores['coverage_pct']==100
    assert scores['pearson_r2']==pytest.approx(1)
    assert scores['rmse_m']==pytest.approx(0.1)
    assert scores['bias_m']==pytest.approx(0.1)
    assert scores['nse']<1
    annual=core.annual_maxima(pair,2020,2020)
    assert bool(annual.loc[0,'usable'])
    assert annual.loc[0,'max_bias_m']==pytest.approx(0.1)


def test_artifact_mask_and_annual_maxima():
    idx=pd.date_range('2001-10-01', '2001-12-31 23:00',freq='h',tz='UTC')
    pair=pd.DataFrame({'observed':np.ones(len(idx)),'cora':np.ones(len(idx))},index=idx)
    clean=h._exclude_wave_artifacts(pair)
    assert not ((clean.index.year==2001)&(clean.index.month==11)).any()
    assert len(clean)==len(pair)-720


def test_summary_missing_and_shared_station():
    ass,_=core.assign_stations(h.load_sites(ROOT/'data'/'study_sites.csv'),h.fallback_stations())
    stats=pd.DataFrame([
        dict(station_id='8651370',kind='WL',pearson_r2=.96,nse=.95,
             rmse_m=.11,bias_m=.02,coverage_pct=40,n_pairs=100,node_offset_km=.5,status='OK')])
    summary=h.build_site_summary(ass,stats)
    assert len(summary)==10
    assert summary[summary.site_id=='09'].iloc[0]['WL_r2']==pytest.approx(.96)
    assert pd.isna(summary[summary.site_id=='09'].iloc[0]['HS_r2'])


def test_chart_export(tmp_path):
    idx=pd.date_range('2020-01-01',periods=5000,freq='h',tz='UTC')
    obs=np.sin(np.arange(len(idx))/12.0)+0.1*np.sin(np.arange(len(idx))/190)
    pair=pd.DataFrame({'observed':obs,'cora':obs*.98+.02},index=idx)
    ann=core.annual_maxima(pair,2020,2020)
    ev=core.observed_events(pair)
    figs={
        'timeseries':h.timeseries_figure(pair,'WL'),
        'events':h.event_figure(ev,'WL'),
        'scatter':h.scatter_figure(pair,'WL'),
        'annual':h.annual_figure(ann,'WL'),
    }
    assert len(figs)==4
    run=tmp_path/'run'
    for sub in ('paired_hourly','annual_maxima','observed_events'):
        (run/sub).mkdir(parents=True)
    pair.to_csv(run/'paired_hourly'/'WL_8651370.csv',index_label='time_utc')
    ann.to_csv(run/'annual_maxima'/'WL_8651370.csv',index=False)
    ev.to_csv(run/'observed_events'/'WL_8651370.csv',index=False)
    pd.DataFrame([dict(station_id='8651370',kind='WL',status='OK')]).to_csv(run/'station_skill.csv',index=False)
    zipdata=h.create_results_zip(run,'WL','8651370',figs)
    zpath=tmp_path/'out.zip';zpath.write_bytes(zipdata)
    with zipfile.ZipFile(zpath) as z:
        names=z.namelist()
        assert 'charts/annual.html' in names
        assert 'paired_hourly/WL_8651370.csv' in names
        assert 'station_skill.csv' in names
    stored=h.stored_result(run,'WL','8651370')
    assert len(stored[0])==len(pair)


def test_invalid_sites_csv():
    with pytest.raises(ValueError,match='requires columns'):
        h.load_sites(ROOT/'data'/'study_sites.csv',b'site_id,foo\n1,bar\n')


def test_import_own_export_and_mismatched_period(tmp_path):
    from io import BytesIO
    import zipfile
    run=tmp_path/'source'
    (run/'paired_hourly').mkdir(parents=True)
    idx=pd.date_range('2020-01-01',periods=10,freq='h',tz='UTC')
    pd.DataFrame({'observed':[0.5]*10,'cora':[0.6]*10},index=idx).to_csv(
        run/'paired_hourly'/'WL_8651370.csv',index_label='time_utc')
    pd.DataFrame([dict(station_id='8651370',kind='WL',years='2020-2022',
                       status='OK',n_pairs=10,exclude_problem_wave_months=True)]).to_csv(
        run/'station_skill.csv',index=False)
    data=h.create_results_zip(run,'WL','8651370')
    target=tmp_path/'dest'
    assert h.import_precomputed_zip(data,target,2020,2022,True)==1
    assert (target/'paired_hourly'/'WL_8651370.csv').exists()
    with pytest.raises(ValueError,match='years differ'):
        h.import_precomputed_zip(data,tmp_path/'other',2010,2012,True)
