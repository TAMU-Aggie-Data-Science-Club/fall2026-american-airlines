"""History-only seasonal pooling and conservative segment-level selection."""
import numpy as np
import pandas as pd
from selection import completed_origins

def avg(x,axis=0):
    n=np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x,axis=axis),n,out=np.full_like(n,np.nan,dtype=float),where=n>0)

def profile_forecast(y,dates,meta,cut,cfg,spec):
    hist=y[:cut+1];recent=avg(hist[-12:]);profiles=[]
    for month in range(1,13):
        rows=hist[dates[:cut+1].month==month];mask=np.isfinite(rows)
        ranks=np.flip(np.cumsum(np.flip(mask,axis=0),axis=0),axis=0)
        p=avg(np.where(mask&(ranks<=spec['years']),rows,np.nan))
        profiles.append(np.where(np.isfinite(p),p,recent))
    profile=np.array(profiles);level=avg(profile);own=np.divide(profile,level,out=np.ones_like(profile),where=level>0)
    pooled=[]
    endpoints=['origin_state_group','destination_state_group'] if spec['pool']=='state' else ['domestic']
    for group in endpoints:
        codes,labels=pd.factorize(meta[group].fillna('UNKNOWN'));sums=np.array([np.bincount(codes,weights=np.nan_to_num(row,nan=0),minlength=len(labels)) for row in profile])
        center=sums.mean(axis=0);ratios=np.divide(sums,center,out=np.ones_like(sums),where=center>0)
        pooled.append(ratios[:,codes])
    group_profile=np.mean(pooled,axis=0)
    confidence=np.nan_to_num(recent,nan=0)/(np.nan_to_num(recent,nan=0)+spec['prior_volume']+1e-9)
    seasonal=own*confidence+group_profile*(1-confidence)
    prev=avg(hist[-24:-12]) if len(hist)>12 else np.full_like(recent,np.nan)
    growth=np.divide(recent,prev,out=np.ones_like(recent),where=prev>0)-1
    growth=np.clip(np.nan_to_num(growth,nan=0),-.15,.15)
    live=np.isfinite(hist[-cfg['active_lookback_months']:]).any(axis=0);out=[]
    for h in cfg['horizons']:
        month=(dates[cut]+pd.DateOffset(months=h)).month
        p=recent*seasonal[month-1]*(1+growth)**(h/12*spec['trend_weight']);p=np.maximum(p,0);p[~live]=np.nan
        if not np.isfinite(p[live]).all():raise ValueError('Invalid profile forecast')
        out.append(p)
    return np.array(out)

def proxy_origins(y,dates,cut,cfg):
    result=[]
    for c in range(cut+1):
        if dates[c].month not in [1,4,7,10] or dates[c].year in cfg['excluded_years'] or c+12>cut:continue
        if np.isfinite(y[:c+1]).any(axis=1).sum()<12:continue
        if np.isfinite(y[c+np.arange(1,13)]).any(axis=1).all():result.append(c)
    return result[-cfg.get('proxy_max_windows',4):]

def select_proxy(y,dates,cut,cfg,cache):
    windows=proxy_origins(y,dates,cut,cfg);best='seasonal_naive';rows=[]
    audit={'proxy_completed_windows':len(windows),'proxy_selection_origins':[str(dates[c].date()) for c in windows],
           'proxy_latest_target':str(dates[windows[-1]+12].date()) if windows else None,'proxy_evidence_horizons':list(range(1,13))}
    if not windows:return best,audit,rows
    families=cfg.get('proxy_candidate_families',['seasonal','level_scaled','growth','profile'])
    names=[s['name'] for s in cfg['candidates'] if s['family'] in families];names+=['blend_'+n for n in names if n!='seasonal_naive']
    truth=np.concatenate([y[c+np.arange(1,13)] for c in windows]);valid=np.isfinite(truth)
    def losses(p):return np.where(valid,np.abs(np.nan_to_num(p,nan=0)-truth),0).sum(axis=1)
    baseline=losses(np.concatenate([cache[(c,'seasonal_naive')] for c in windows]));denom=np.nansum(truth)
    baseline_score=baseline.sum()/denom*100;best_score=baseline_score
    for name in names:
        if not all((c,name) in cache for c in windows):continue
        loss=losses(np.concatenate([cache[(c,name)] for c in windows]));wape=loss.sum()/denom*100
        wins=float(np.mean(loss<baseline));gain=1-wape/baseline_score if baseline_score>0 else 0
        eligible=name=='seasonal_naive' or (gain>=cfg.get('proxy_min_relative_gain',.05) and wins>=cfg.get('proxy_min_month_win_fraction',.6))
        rows.append({'candidate':name,'proxy_wape_pct':wape,'relative_gain':gain,'month_win_fraction':wins,'eligible':eligible})
        if eligible and wape<best_score:best=name;best_score=wape
    audit['proxy_selected_model']=best;audit['proxy_selection_wape_pct']=best_score
    return best,audit,rows

