"""Second-year demand: 12 monthly forecasts at calendar horizons 13 through 24."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse, json, warnings, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from selection import completed_origins, select_at_origin
from market_methods import profile_forecast, proxy_origins, select_proxy

FEATURES=['log_same_month_demand','log_recent_mean','log_latest_demand','recent_growth',
          'month_sin','month_cos','horizon','historical_months','recent_months',
          'domestic','distance_km','origin_latitude','destination_latitude']

def mean(a,axis=0):
    count=np.isfinite(a).sum(axis=axis);total=np.nansum(a,axis=axis)
    return np.divide(total,count,out=np.full_like(total,np.nan,dtype=float),where=count>0)

def seasonal(y,dates,cut,month,years):
    hist=y[:cut+1];idx=np.flatnonzero(dates[:cut+1].month==month)
    values=hist[idx];mask=np.isfinite(values)
    # Select latest N observed values separately for every market.
    ranks=np.flip(np.cumsum(np.flip(mask,axis=0),axis=0),axis=0)
    selected=np.where(mask&(ranks<=years),values,np.nan)
    s=mean(selected)
    return np.where(np.isfinite(s),s,mean(hist[-12:]))

def active(y,cut,cfg):return np.isfinite(y[max(0,cut-cfg['active_lookback_months']+1):cut+1]).any(axis=0)

def features(y,dates,meta,cut,h,cfg):
    hist=y[:cut+1];recent=hist[-12:];prev=hist[-24:-12]
    rm=mean(recent);pm=mean(prev) if len(prev) else np.full(y.shape[1],np.nan)
    growth=np.divide(rm,pm,out=np.full_like(rm,np.nan),where=pm>0)-1
    last_idx=np.where(np.isfinite(hist),np.arange(len(hist))[:,None],-1).max(axis=0)
    last=hist[np.maximum(last_idx,0),np.arange(y.shape[1])];last[last_idx<0]=np.nan
    target=dates[cut]+pd.DateOffset(months=int(h));s=seasonal(y,dates,cut,target.month,3)
    scale=np.maximum(np.nan_to_num(s,nan=0),cfg['scale_floor'])
    f=np.column_stack([np.log1p(scale),np.log1p(rm),np.log1p(last),growth,
        np.full(y.shape[1],np.sin(2*np.pi*target.month/12)),np.full(y.shape[1],np.cos(2*np.pi*target.month/12)),
        np.full(y.shape[1],h),np.isfinite(hist).sum(axis=0),np.isfinite(recent).sum(axis=0),
        meta.domestic,meta.distance_km,meta.origin_latitude,meta.destination_latitude])
    return f,scale

def make_training(y,dates,meta,cut,cfg):
    rng=np.random.default_rng(cfg['seed']);pairs=[]
    for origin in range(cfg['ml_min_training_origin_months']-1,cut+1,cfg['train_origin_stride']):
        if not np.isfinite(y[origin]).any():continue
        for h in cfg['horizons']:
            if origin+h<=cut and np.isfinite(y[origin+h]).any():pairs.append((origin,h))
    if not pairs:return None
    per=max(1,cfg['training_pair_sample']//len(pairs));xs=[];ts=[]
    for origin,h in pairs:
        eligible=np.flatnonzero(active(y,origin,cfg)&np.isfinite(y[origin+h]))
        if not len(eligible):continue
        ids=rng.choice(eligible,min(per,len(eligible)),replace=False)
        f,s=features(y,dates,meta,origin,h,cfg)
        xs.append(f[ids]);ts.append(np.log1p(y[origin+h,ids])-np.log1p(s[ids]))
    if not xs:return None
    return np.concatenate(xs),np.concatenate(ts)

def fit(spec,training,cfg):
    if spec['family'] not in ['rf','boost']:return None
    if training is None or len(training[1])<cfg['minimum_ml_pairs']:return None
    params={k:v for k,v in spec.items() if k not in ['name','family']}
    if spec['family']=='rf':
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestRegressor(n_jobs=2,random_state=cfg['seed'],**params))
    else:model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),VarianceThreshold(),HistGradientBoostingRegressor(random_state=cfg['seed'],early_stopping=False,**params))
    model.fit(*training);return model

def predict(y,dates,meta,cut,cfg,spec,model=None):
    if spec['family']=='profile':return profile_forecast(y,dates,meta,cut,cfg,spec)
    live=active(y,cut,cfg);pred=[];family=spec['family']
    for h in cfg['horizons']:
        target=dates[cut]+pd.DateOffset(months=h)
        s=seasonal(y,dates,cut,target.month,spec.get('years',1))
        if family=='level_scaled':
            rm=mean(y[max(0,cut-11):cut+1]);old=mean(y[max(0,cut-35):cut+1]);ratio=np.divide(rm,old,out=np.ones_like(rm),where=old>0)
            s=s*(1+spec['adjustment']*(ratio-1))
        elif family=='growth':
            rm=mean(y[max(0,cut-11):cut+1]);prev=y[max(0,cut-23):max(0,cut-11)]
            pm=mean(prev) if len(prev) else np.full(y.shape[1],np.nan)
            g=np.divide(rm,pm,out=np.ones_like(rm),where=pm>0)-1;g=np.nan_to_num(g,nan=0)
            s=s*(1+np.clip(g,-spec['growth_cap'],spec['growth_cap']))**(h/12*spec['damping'])
        elif family in ['rf','boost'] and model is not None:
            f,scale=features(y,dates,meta,cut,h,cfg)
            s=np.full(y.shape[1],np.nan)
            s[live]=np.expm1(model.predict(f[live])+np.log1p(scale[live]))
        s=np.maximum(s,0);s[~live]=np.nan
        if not np.isfinite(s[live]).all():raise ValueError('Nonfinite forecast for active market')
        pred.append(s)
    return np.array(pred)

def score(actual,pred):
    mask=np.isfinite(actual);p=np.nan_to_num(pred,nan=0);err=np.where(mask,p-actual,0)
    denom=np.nansum(actual)
    return {'market_wape_pct':float(np.abs(err).sum()/denom*100),'market_mae':float(np.abs(err).sum()/mask.sum()),
        'bias_pct':float(err.sum()/denom*100),'observed_market_months':int(mask.sum()),
        'unforecast_actual_demand_pct':float(np.where(mask&~np.isfinite(pred),actual,0).sum()/denom*100),
        'unlabelled_forecast_market_months':int((~mask&np.isfinite(pred)).sum()),
        'aggregate_wape_pct':float(np.abs(np.nansum(pred,axis=1)-np.nansum(actual,axis=1)).sum()/denom*100)}

def run(root):
    root=Path(root);cfg=json.loads((root/'config.json').read_text());out=root/'data/processed/results';out.mkdir(parents=True,exist_ok=True)
    if cfg['horizons']!=list(range(13,25)):raise ValueError('This model requires horizons 13 through 24')
    names=[s['name'] for s in cfg['candidates']]
    if len(set(names))!=len(names) or 'seasonal_naive' not in names or any(n.startswith('blend_') for n in names):raise ValueError('Candidate names must be unique, include seasonal_naive and not start with blend_')
    data=np.load(root/'data/processed/panel.npz',allow_pickle=False);dates=pd.DatetimeIndex(data['dates']);values=data['values'];ids=data['markets']
    meta=pd.read_csv(root/'data/processed/markets.csv').set_index('market').reindex(ids).reset_index()
    cap=pd.Timestamp(cfg['as_of']).to_period('M').to_timestamp()-pd.offsets.MonthBegin(1)
    if cfg['last_complete_month']:cap=min(cap,pd.Timestamp(cfg['last_complete_month']))
    use=dates<=cap;dates=dates[use];raw=values[use];y=raw.copy();y[dates.year.isin(cfg['excluded_years'])]=np.nan
    origins=[dates.get_loc(pd.Timestamp(d)) for d in cfg['backtest_origins']]
    for c in origins:
        if c+24>=len(dates) or dates[c].year in cfg['excluded_years']:raise ValueError('Invalid backtest origin')
        if not np.isfinite(y[c+np.array(cfg['horizons'])]).any(axis=1).all():raise ValueError('Missing/COVID evaluation month')
    work_origins=sorted(set(origins).union(completed_origins(y,dates,len(dates)-1,cfg),*[completed_origins(y,dates,c,cfg) for c in origins]))
    # Proxy selection uses shorter completed outcomes only when year-two evidence
    # is unavailable. The evaluation target remains horizons 13-24 throughout.
    proxy_decisions=[c for c in origins+[len(dates)-1] if len(completed_origins(y,dates,c,cfg))<cfg['selection_min_windows']]
    proxy_cuts=sorted(set().union(*[proxy_origins(y,dates,c,cfg) for c in proxy_decisions]))
    proxy_cache={};proxy_cfg={**cfg,'horizons':list(range(1,13))}
    proxy_specs=[s for s in cfg['candidates'] if s['family'] in cfg.get('proxy_candidate_families',['seasonal','level_scaled','growth','profile'])]
    for c in proxy_cuts:
        print('Completed proxy window origin',dates[c].date(),flush=True)
        for spec in proxy_specs:proxy_cache[(c,spec['name'])]=predict(y,dates,meta,c,proxy_cfg,spec)
        for spec in proxy_specs:
            if spec['name']!='seasonal_naive':proxy_cache[(c,'blend_'+spec['name'])]=(proxy_cache[(c,spec['name'])]+proxy_cache[(c,'seasonal_naive')])/2
    scores=[];monthly=[];all_predictions={};last_models={};fail=[];timings=[];trained_counts=[]
    for cut in work_origins:
        print('Backtest origin',dates[cut].date(),flush=True);start=time.time();training=make_training(y,dates,meta,cut,cfg)
        print('Training pairs',0 if training is None else len(training[1]),flush=True)
        actual=y[cut+np.array(cfg['horizons'])]
        for spec in cfg['candidates']:
            print('  Fit',spec['name'],flush=True);st=time.time()
            try:
                model=fit(spec,training,cfg);p=predict(y,dates,meta,cut,cfg,spec,model)
                all_predictions[(cut,spec['name'])]=p
                scores.append({'origin':str(dates[cut].date()),'candidate':spec['name'],**score(actual,p),
                    'cold_start_fallback':spec['family'] in ['rf','boost'] and model is None})
                for i,h in enumerate(cfg['horizons']):monthly.append({'origin':str(dates[cut].date()),'target':str(dates[cut+h].date()),'horizon':h,'candidate':spec['name'],'actual':float(np.nansum(actual[i])),'forecast':float(np.nansum(p[i]))})
                if cut==origins[-1]:last_models[spec['name']]=model
            except Exception as e:fail.append({'origin':str(dates[cut].date()),'candidate':spec['name'],'error':str(e)})
            timings.append({'origin':str(dates[cut].date()),'candidate':spec['name'],'seconds':time.time()-st})
        trained_counts.append({'origin':str(dates[cut].date()),'training_pairs':0 if training is None else len(training[1]),'seconds':time.time()-start})
    # Candidates must succeed in every fold; equal-weight ensembles are fixed rules.
    valid=[s['name'] for s in cfg['candidates'] if all((c,s['name']) in all_predictions for c in origins)]
    for name in [s['name'] for s in cfg['candidates']]:
        if name=='seasonal_naive':continue
        blend='blend_'+name
        for cut in work_origins:
            if (cut,name) not in all_predictions:continue
            p=(all_predictions[(cut,name)]+all_predictions[(cut,'seasonal_naive')])/2
            all_predictions[(cut,blend)]=p;actual=y[cut+np.array(cfg['horizons'])]
            scores.append({'origin':str(dates[cut].date()),'candidate':blend,**score(actual,p),'cold_start_fallback':False})
            for i,h in enumerate(cfg['horizons']):monthly.append({'origin':str(dates[cut].date()),'target':str(dates[cut+h].date()),'horizon':h,'candidate':blend,'actual':float(np.nansum(actual[i])),'forecast':float(np.nansum(p[i]))})
    candidates=valid+['blend_'+n for n in valid if n!='seasonal_naive'];rank=[]
    actual_all=np.concatenate([y[c+np.array(cfg['horizons'])] for c in origins])
    for name in candidates:
        p=np.concatenate([all_predictions[(c,name)] for c in origins]);rank.append({'candidate':name,**score(actual_all,p)})
    ranking=pd.DataFrame(rank).sort_values('market_wape_pct')
    # Nested, as-of selection: future evaluation errors cannot choose past winners.
    audits=[];proxy_score_rows=[]
    for cut in origins:
        selected,audit,_=select_at_origin(y,dates,cut,cfg,all_predictions)
        if audit['completed_windows']<cfg['selection_min_windows']:
            selected,proxy_audit,proxy_rows=select_proxy(y,dates,cut,cfg,proxy_cache);audit.update(proxy_audit)
            audit['reason']='Completed one-year proxy evidence; insufficient year-two windows' if proxy_audit['proxy_completed_windows'] else 'No completed selection evidence; predetermined baseline'
            proxy_score_rows += [{'decision_origin':str(dates[cut].date()),**r} for r in proxy_rows]
        if (cut,selected) not in all_predictions:
            selected='seasonal_naive';audit['reason']='Candidate failed to fit at origin; baseline fallback'
        audit['selected_model']=selected;audits.append(audit)
        p=all_predictions[(cut,selected)];all_predictions[(cut,'nested_selected')]=p
        actual=y[cut+np.array(cfg['horizons'])]
        scores.append({'origin':str(dates[cut].date()),'candidate':'nested_selected',**score(actual,p),'cold_start_fallback':False})
        for i,h in enumerate(cfg['horizons']):monthly.append({'origin':str(dates[cut].date()),'target':str(dates[cut+h].date()),'horizon':h,'candidate':'nested_selected','actual':float(np.nansum(actual[i])),'forecast':float(np.nansum(p[i]))})
    nested_metrics=score(actual_all,np.concatenate([all_predictions[(c,'nested_selected')] for c in origins]))
    winner,final_audit,selection_rows=select_at_origin(y,dates,len(dates)-1,cfg,all_predictions)
    if final_audit['completed_windows']<cfg['selection_min_windows']:
        winner,proxy_audit,proxy_rows=select_proxy(y,dates,len(dates)-1,cfg,proxy_cache);final_audit.update(proxy_audit)
        final_audit['reason']='Completed one-year proxy evidence; insufficient year-two windows' if proxy_audit['proxy_completed_windows'] else 'No completed selection evidence; predetermined baseline'
        proxy_score_rows += [{'decision_origin':str(dates[-1].date()),**r} for r in proxy_rows]
    final_audit['selected_model']=winner;final_audit['purpose']='current forecast';audits.append(final_audit)
    (out/'selection_audit.json').write_text(json.dumps(audits,indent=2))
    pd.DataFrame(selection_rows).to_csv(out/'current_selection_scores.csv',index=False)
    pd.DataFrame(proxy_score_rows).to_csv(out/'proxy_selection_scores.csv',index=False)
    pd.DataFrame([{'procedure':'as_of_selection',**nested_metrics},{'procedure':'seasonal_naive',**ranking[ranking.candidate=='seasonal_naive'].iloc[0].drop('candidate').to_dict()}]).to_csv(out/'nested_backtest_summary.csv',index=False)
    print('Selected',winner,flush=True)
    ranking.to_csv(out/'model_comparison.csv',index=False)
    for frame,filename in [(pd.DataFrame(scores),'backtest_scores.csv'),(pd.DataFrame(monthly),'backtest_monthly_totals.csv')]:
        frame['window_role']=np.where(frame.origin.isin(cfg['backtest_origins']),'outer_evaluation','selection_only')
        frame.to_csv(out/filename,index=False)
    pd.DataFrame(timings).to_csv(out/'training_times.csv',index=False);pd.DataFrame(trained_counts).to_csv(out/'training_pair_counts.csv',index=False)
    (out/'candidate_failures.json').write_text(json.dumps(fail,indent=2))
    # Market error diagnostics include every target-labelled market, including unseen ones.
    error_abs=np.zeros(len(meta));denom=np.zeros(len(meta));obs_count=np.zeros(len(meta));total_bias=np.zeros(len(meta))
    for c in origins:
        a=y[c+np.array(cfg['horizons'])];p=np.nan_to_num(all_predictions[(c,'nested_selected')],nan=0);mask=np.isfinite(a)
        e=np.where(mask,p-a,0);error_abs+=np.abs(e).sum(axis=0);denom+=np.nansum(a,axis=0);obs_count+=mask.sum(axis=0);total_bias+=e.sum(axis=0)
    market_metrics=meta.copy();market_metrics['backtest_actual']=denom;market_metrics['backtest_absolute_error']=error_abs;market_metrics['backtest_bias']=total_bias;market_metrics['observed_market_months']=obs_count
    market_metrics['wape_pct']=np.divide(error_abs,denom,out=np.full_like(denom,np.nan),where=denom>0)*100
    market_metrics.to_csv(out/'market_backtest_metrics.csv',index=False)
    for endpoint in ['origin','destination']:
        g=market_metrics.groupby(endpoint+'_state_group')[['backtest_actual','backtest_absolute_error','backtest_bias','observed_market_months']].sum()
        g['market_wape_pct']=g.backtest_absolute_error/g.backtest_actual*100;g.to_csv(out/(endpoint+'_state_backtest_metrics.csv'))
        codes,groups=pd.factorize(meta[endpoint+'_state_group']);state_rows=[]
        for c in origins:
            for j,h in enumerate(cfg['horizons']):
                a=np.bincount(codes,weights=np.nan_to_num(y[c+h],nan=0),minlength=len(groups))
                p=np.bincount(codes,weights=np.nan_to_num(all_predictions[(c,'nested_selected')][j],nan=0),minlength=len(groups))
                for k,group in enumerate(groups):state_rows.append({'origin':str(dates[c].date()),'target':str(dates[c+h].date()),'state_group':group,'observed_total':a[k],'forecast_total':p[k]})
        state=pd.DataFrame(state_rows);state.to_csv(out/(endpoint+'_state_backtest_monthly.csv'),index=False)
        state['absolute_error']=abs(state.forecast_total-state.observed_total)
        ss=state.groupby('state_group')[['absolute_error','observed_total']].sum();ss['state_aggregate_wape_pct']=ss.absolute_error/ss.observed_total*100;ss.to_csv(out/(endpoint+'_state_aggregate_scores.csv'))
    # Forecast all eligible markets at the final origin.
    latest=len(dates)-1;underlying=winner.removeprefix('blend_');spec=next(s for s in cfg['candidates'] if s['name']==underlying)
    train=make_training(y,dates,meta,latest,cfg) if spec['family'] in ['rf','boost'] else None
    model=fit(spec,train,cfg);p=predict(y,dates,meta,latest,cfg,spec,model)
    if winner.startswith('blend_'):p=(p+predict(y,dates,meta,latest,cfg,next(s for s in cfg['candidates'] if s['name']=='seasonal_naive')))/2
    live=active(y,latest,cfg);idx=np.flatnonzero(live);frames=[]
    recent=y[max(0,latest-11):latest+1];counts=np.isfinite(recent).sum(axis=0)
    for j,h in enumerate(cfg['horizons']):
        f=meta.iloc[idx].copy();f['date']=dates[latest]+pd.DateOffset(months=h);f['horizon']=h;f['forecast_passengers']=p[j,idx];f['recent_observed_months']=counts[idx];frames.append(f)
    forecast=pd.concat(frames,ignore_index=True);forecast.to_csv(out/'market_year2_monthly.csv',index=False)
    annual=forecast.groupby(['market','origin','destination','origin_state_group','destination_state_group'],dropna=False).forecast_passengers.sum().reset_index(name='year2_forecast')
    prior=np.nansum(recent,axis=0);prior[counts<12]=np.nan
    annual['last_12_month_actual']=annual.market.map(dict(zip(meta.market,prior)));annual['change_vs_last_12_pct']=(annual.year2_forecast/annual.last_12_month_actual-1)*100
    annual['change_vs_last_12_pct']=annual.change_vs_last_12_pct.replace([np.inf,-np.inf],np.nan)
    annual=annual.sort_values('year2_forecast',ascending=False);annual.to_csv(out/'market_year2_annual.csv',index=False)
    total=forecast.groupby(['date','horizon']).forecast_passengers.sum().reset_index();total.to_csv(out/'total_year2_monthly.csv',index=False)
    for endpoint in ['origin','destination']:
        forecast.groupby(['date',endpoint+'_state_group'],dropna=False).forecast_passengers.sum().reset_index().to_csv(out/(endpoint+'_state_year2_monthly.csv'),index=False)
        annual.groupby(endpoint+'_state_group',dropna=False).year2_forecast.sum().sort_values(ascending=False).to_csv(out/(endpoint+'_state_year2_annual.csv'))
        forecast.groupby(['date',endpoint],dropna=False).forecast_passengers.sum().reset_index().to_csv(out/(endpoint+'_airport_year2_monthly.csv'),index=False)
    # Feature permutation on the last development window for the best ML candidate.
    ml_names=[s['name'] for s in cfg['candidates'] if s['family'] in ['rf','boost'] and last_models.get(s['name']) is not None]
    diagnostic_name=next((n for n in ranking.candidate if n in ml_names),None);importance=[]
    if diagnostic_name:
        cut=origins[-1];m=last_models[diagnostic_name];rng=np.random.default_rng(cfg['seed']);xs=[];actuals=[];scales=[]
        for h in cfg['horizons']:
            eligible=np.flatnonzero(active(y,cut,cfg)&np.isfinite(y[cut+h]));sample=rng.choice(eligible,min(1500,len(eligible)),replace=False)
            x,s=features(y,dates,meta,cut,h,cfg);xs.append(x[sample]);scales.append(s[sample]);actuals.append(y[cut+h,sample])
        x=np.concatenate(xs);s=np.concatenate(scales);a=np.concatenate(actuals)
        def loss(v):return np.abs(np.maximum(0,np.expm1(m.predict(v)+np.log1p(s)))-a).sum()/a.sum()*100
        base=loss(x)
        for k,name in enumerate(FEATURES):
            xp=x.copy();xp[:,k]=xp[rng.permutation(len(x)),k];importance.append({'feature':name,'increase_in_sample_wape_percentage_points':loss(xp)-base,'model':diagnostic_name,'evaluation':'last development fold; associations, not causality'})
        pd.DataFrame(importance).sort_values('increase_in_sample_wape_percentage_points',ascending=False).to_csv(out/'feature_permutation_importance.csv',index=False)
    summary={'status':'As-of nested historical evaluation; data previously explored; not prospective proof','horizons':cfg['horizons'],'origin':str(dates[latest].date()),'forecast_start':str(total.date.min().date()),'forecast_end':str(total.date.max().date()),'year2_total':float(total.forecast_passengers.sum()),'selected_model':winner,'selected_spec':spec,'selected_development_metrics':ranking[ranking.candidate==winner].iloc[0].to_dict(),'nested_backtest_metrics':nested_metrics,'baseline_metrics':ranking[ranking.candidate=='seasonal_naive'].iloc[0].to_dict(),'evaluation_series':'nested_selected','final_selection_audit':final_audit,'evaluated_candidates':len(ranking),'backtest_windows':len(origins),'active_markets':int(live.sum()),'inactive_known_markets':int((~live).sum()),'excluded_years':cfg['excluded_years'],'feature_diagnostic_model':diagnostic_name,'final_training_pairs':0 if train is None else len(train[1]),'blend_weight':.5 if winner.startswith('blend_') else None}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with open(out/'model.pkl','wb') as f:pickle.dump({'model':model,'spec':spec,'winner':winner,'config':cfg,'feature_names':FEATURES,'origin':dates[latest]},f)
    # Final-window chart clearly separates historical evaluation from future estimates.
    chosen=pd.DataFrame(monthly);chosen=chosen[(chosen.candidate=='nested_selected')&(chosen.origin==str(dates[origins[-1]].date()))]
    fig,axs=plt.subplots(2,1,figsize=(12,8));axs[0].plot(pd.to_datetime(chosen.target),chosen.actual/1e6,label='Observed');axs[0].plot(pd.to_datetime(chosen.target),chosen.forecast/1e6,label='Model chosen as of '+str(dates[origins[-1]].date()));axs[0].set_title('Latest historical second-year window — nested as-of selection');axs[0].legend();axs[0].set_ylabel('OD_PAX (millions)')
    axs[1].plot(total.date,total.forecast_passengers/1e6,marker='o',color='#d87926');axs[1].set_title('Second year ahead: '+total.date.min().strftime('%b %Y')+'–'+total.date.max().strftime('%b %Y'));axs[1].set_ylabel('Forecast OD_PAX (millions)');axs[1].grid(alpha=.2);fig.tight_layout();fig.savefig(out/'year2_forecast.png',dpi=160);plt.close(fig)
    top=annual.head(15).iloc[::-1];fig,ax=plt.subplots(figsize=(10,7));ax.barh(top.market,top.year2_forecast/1e6,color='#25485c');ax.set(xlabel='Second-year forecast OD_PAX (millions)',title='Largest directional markets by forecast demand');fig.tight_layout();fig.savefig(out/'top_markets.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).parent));a=p.parse_args();run(a.root)
