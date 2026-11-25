import unittest
import tempfile,zipfile,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from model import seasonal, active, features, make_training, predict, score, fit
from selection import completed_origins, select_at_origin
from evaluate_issue import evaluate
from market_methods import profile_forecast, proxy_origins, select_proxy

class YearTwoTests(unittest.TestCase):
    def setUp(self):
        self.dates=pd.date_range('2019-01-01',periods=96,freq='MS')
        self.y=np.column_stack([100+np.arange(96),200+np.arange(96),np.zeros(96)]).astype(float)
        self.y[self.dates.year.isin([2020,2021])]=np.nan
        self.meta=pd.DataFrame({'domestic':[1,1,0],'distance_km':[100,200,300],'origin_latitude':[30,40,50],'destination_latitude':[31,41,51]})
        self.cfg={'seed':42,'training_pair_sample':1000,'train_origin_stride':3,'horizons':list(range(13,25)), 'active_lookback_months':12,'scale_floor':10,'ml_min_training_origin_months':12}
    def test_full_second_year(self):
        origin=pd.Timestamp('2026-07-01');dates=[origin+pd.DateOffset(months=h) for h in self.cfg['horizons']]
        self.assertEqual(len(dates),12);self.assertEqual(dates[0],pd.Timestamp('2027-08-01'));self.assertEqual(dates[-1],pd.Timestamp('2028-07-01'))
    def test_real_zero_is_preserved(self):
        p=predict(self.y,self.dates,self.meta,66,self.cfg,{'family':'seasonal','years':3})
        np.testing.assert_equal(p[:,2],0)
    def test_unseen_market_is_not_a_zero_actual(self):
        y=self.y.copy();y[:67,1]=np.nan
        p=predict(y,self.dates,self.meta,66,self.cfg,{'family':'seasonal','years':3})
        self.assertTrue(np.isnan(p[:,1]).all())
        s=score(np.array([[10.,np.nan,20.]]),np.array([[8.,99.,np.nan]]))
        self.assertAlmostEqual(s['market_wape_pct'],22/30*100)
        self.assertEqual(s['observed_market_months'],2)
        self.assertEqual(s['unlabelled_forecast_market_months'],1)
    def test_future_cannot_change_features_or_training(self):
        cut=66;y=self.y.copy();y[cut+1:]=999999
        a=features(self.y,self.dates,self.meta,cut,24,self.cfg);b=features(y,self.dates,self.meta,cut,24,self.cfg)
        for x,z in zip(a,b):np.testing.assert_allclose(x,z,equal_nan=True)
        a=make_training(self.y,self.dates,self.meta,cut,self.cfg);b=make_training(y,self.dates,self.meta,cut,self.cfg)
        for x,z in zip(a,b):np.testing.assert_allclose(x,z,equal_nan=True)
    def test_seasonal_mean_ignores_excluded_years(self):
        cut=66;s=seasonal(self.y,self.dates,cut,7,5)
        eligible=self.y[:cut+1][self.dates[:cut+1].month==7]
        np.testing.assert_allclose(s,np.nanmean(eligible,axis=0))
        self.assertEqual(np.isnan(self.y[self.dates.year.isin([2020,2021])]).sum(),72)
    def test_inactive_markets_do_not_leak_into_forecast(self):
        y=self.y.copy();y[55:67,1]=np.nan
        self.assertFalse(active(y,66,self.cfg)[1])
    def test_boost_handles_constant_and_missing_columns(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(100,13));x[:,3]=np.nan;x[:,6]=13
        cfg={**self.cfg,'minimum_ml_pairs':20}
        model=fit({'name':'test','family':'boost','max_iter':5,'max_leaf_nodes':3},(x,x[:,0]*.1),cfg)
        self.assertTrue(np.isfinite(model.predict(x)).all())
    def selector_config(self):
        return {**self.cfg,'excluded_years':[2020,2021],'selection_min_windows':2,'selection_max_windows':7,'selection_min_relative_gain':.02,'selection_min_win_fraction':.6,'candidates':[{'name':'seasonal_naive'},{'name':'seasonal_2'}]}
    def test_selection_cannot_see_future_results(self):
        cfg=self.selector_config();cut=66;preds={}
        for c in [36,39,42,45]:
            a=self.y[c+np.array(cfg['horizons'])]
            preds[(c,'seasonal_naive')]=a*1.2;preds[(c,'seasonal_2')]=a*1.05
        selected,audit,_=select_at_origin(self.y,self.dates,cut,cfg,preds)
        changed=self.y.copy();changed[cut+1:]=999999
        preds[(45,'seasonal_2')]=np.full((12,3),-999999)
        other,audit2,_=select_at_origin(changed,self.dates,cut,cfg,preds)
        self.assertEqual(selected,'seasonal_2');self.assertEqual(other,selected);self.assertEqual(audit,audit2)
        self.assertLessEqual(pd.Timestamp(audit['latest_selection_target']),self.dates[cut])
    def test_selection_falls_back_without_completed_windows(self):
        name,audit,_=select_at_origin(self.y,self.dates,54,self.selector_config(),{})
        self.assertEqual(name,'seasonal_naive');self.assertEqual(audit['completed_windows'],0)
    def test_selection_rejects_unproven_improvement(self):
        cfg=self.selector_config();preds={}
        for c in completed_origins(self.y,self.dates,66,cfg):
            a=self.y[c+np.array(cfg['horizons'])]
            preds[(c,'seasonal_naive')]=a*1.2;preds[(c,'seasonal_2')]=a*1.199
        name,_,_=select_at_origin(self.y,self.dates,66,cfg,preds)
        self.assertEqual(name,'seasonal_naive')
    def make_issue(self,folder):
        payload=b'date,market,forecast_passengers\n2027-08-01,A>B,8\n2027-08-01,B>A,9\n'
        issue=Path(folder)/'issue.zip'
        with zipfile.ZipFile(issue,'w') as z:
            z.writestr('manifest.json',json.dumps({'issue_id':'test','forecast_origin':'2026-07-01','forecast_sha256':hashlib.sha256(payload).hexdigest()}));z.writestr('market_forecast.csv',payload)
        return issue
    def test_prospective_score_waits_for_actuals(self):
        with tempfile.TemporaryDirectory() as tmp:
            issue=self.make_issue(tmp);panel=Path(tmp)/'panel.npz'
            np.savez(panel,dates=['2026-07-01'],markets=['A>B'],values=[[10.]])
            result=evaluate(issue,panel);self.assertEqual(result['status'],'pending actuals');self.assertNotIn('market_wape_pct',result)
    def test_prospective_score_handles_unseen_and_missing_markets(self):
        with tempfile.TemporaryDirectory() as tmp:
            issue=self.make_issue(tmp);panel=Path(tmp)/'panel.npz'
            np.savez(panel,dates=['2027-08-01'],markets=['A>B','B>A','C>D'],values=[[10.,np.nan,30.]])
            result=evaluate(issue,panel);self.assertAlmostEqual(result['market_wape_pct'],80.)
            self.assertEqual(result['forecast_market_months_without_actuals'],1);self.assertEqual(result['unforecast_actual_demand'],30.)
    def test_pooled_profile_ignores_future_and_preserves_zero(self):
        cut=66;y=self.y.copy();y[cut+1:]=999999;meta=self.meta.copy()
        meta['origin_state_group']=['US:TX','US:CA','GB:UNKNOWN'];meta['destination_state_group']=['US:CA','US:TX','US:NY']
        for pool in ['domestic','state']:
            spec={'years':3,'prior_volume':1000,'trend_weight':.25,'pool':pool}
            a=profile_forecast(self.y,self.dates,meta,cut,self.cfg,spec);b=profile_forecast(y,self.dates,meta,cut,self.cfg,spec)
            np.testing.assert_allclose(a,b,equal_nan=True);np.testing.assert_equal(a[:,2],0)
    def test_proxy_selection_cannot_read_future_outcomes(self):
        cfg={**self.selector_config(),'candidates':[{'name':'seasonal_naive','family':'seasonal'},{'name':'seasonal_2','family':'seasonal'}]};cut=66;cache={}
        for c in proxy_origins(self.y,self.dates,cut,cfg):
            a=self.y[c+np.arange(1,13)];cache[(c,'seasonal_naive')]=a*1.2;cache[(c,'seasonal_2')]=a*1.05
        before,audit,_=select_proxy(self.y,self.dates,cut,cfg,cache)
        y=self.y.copy();y[cut+1:]=999999;cache[(cut,'seasonal_2')]=np.zeros((12,3))
        after,audit2,_=select_proxy(y,self.dates,cut,cfg,cache)
        self.assertEqual(before,'seasonal_2');self.assertEqual(before,after);self.assertEqual(audit,audit2)
        self.assertLessEqual(pd.Timestamp(audit['proxy_latest_target']),self.dates[cut])
    def test_proxy_requires_meaningful_improvement(self):
        cfg={**self.selector_config(),'candidates':[{'name':'seasonal_naive','family':'seasonal'},{'name':'seasonal_2','family':'seasonal'}]};cut=66;cache={}
        for c in proxy_origins(self.y,self.dates,cut,cfg):
            a=self.y[c+np.arange(1,13)];cache[(c,'seasonal_naive')]=a*1.2;cache[(c,'seasonal_2')]=a*1.199
        name,_,_=select_proxy(self.y,self.dates,cut,cfg,cache);self.assertEqual(name,'seasonal_naive')

if __name__=='__main__':unittest.main()
