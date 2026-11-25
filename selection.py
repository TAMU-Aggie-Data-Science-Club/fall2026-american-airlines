"""Model selection restricted to outcomes known at the forecast origin."""
import numpy as np
import pandas as pd

def completed_origins(y,dates,decision_cut,cfg):
    result=[];h=np.array(cfg['horizons'])
    for cut in range(decision_cut+1):
        if dates[cut].month not in [1,4,7,10] or cut+h.max()>decision_cut:continue
        if dates[cut].year in cfg['excluded_years']:continue
        if np.isfinite(y[:cut+1]).any(axis=1).sum()<12:continue
        if not np.isfinite(y[cut+h]).any(axis=1).all():continue
        result.append(cut)
    return result[-cfg['selection_max_windows']:]

def select_at_origin(y,dates,decision_cut,cfg,forecasts):
    cuts=completed_origins(y,dates,decision_cut,cfg)
    audit={'decision_origin':str(dates[decision_cut].date()),'completed_windows':len(cuts),
           'latest_selection_target':str(dates[cuts[-1]+max(cfg['horizons'])].date()) if cuts else None,
           'selection_origins':[str(dates[c].date()) for c in cuts]}
    if len(cuts)<cfg['selection_min_windows']:
        return 'seasonal_naive',{**audit,'reason':'Insufficient completed windows; predetermined baseline'},[]
    names=[s['name'] for s in cfg['candidates']]
    names+=['blend_'+n for n in names if n!='seasonal_naive']
    horizon=np.array(cfg['horizons']);rows=[]
    def error(actual,pred):
        valid=np.isfinite(actual)
        return np.where(valid,np.abs(np.nan_to_num(pred,nan=0)-actual),0).sum(),np.nansum(actual)
    baseline=[]
    for c in cuts:
        if (c,'seasonal_naive') not in forecasts:raise ValueError('Missing required baseline forecast')
        baseline.append(error(y[c+horizon],forecasts[(c,'seasonal_naive')]))
    baseline_score=sum(e for e,d in baseline)/sum(d for e,d in baseline)*100
    for name in names:
        if not all((c,name) in forecasts for c in cuts):continue
        e=[]
        for c in cuts:e.append(error(y[c+horizon],forecasts[(c,name)]))
        wape=sum(v for v,d in e)/sum(d for v,d in e)*100
        win_fraction=float(np.mean([v/d<b/bd for (v,d),(b,bd) in zip(e,baseline)]))
        improvement=1-wape/baseline_score if baseline_score>0 else 0
        eligible=name=='seasonal_naive' or (improvement>=cfg['selection_min_relative_gain'] and win_fraction>=cfg['selection_min_win_fraction'])
        rows.append({'candidate':name,'selection_wape_pct':wape,'win_fraction':win_fraction,'relative_gain':improvement,'eligible':eligible})
    best=min([r for r in rows if r['eligible']],key=lambda r:r['selection_wape_pct'])
    return best['candidate'],{**audit,'reason':'Best eligible candidate using already completed windows','selection_wape_pct':best['selection_wape_pct']},rows
