"""Read supplied workbooks locally and cache the monthly market panel."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import load_workbook

def fingerprint(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def prepare(pax,airports,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[];book=load_workbook(pax,read_only=True,data_only=True)
    for sheet in book:
        it=iter(sheet.values);header=next(it)
        cols=['DEP_YEAR','DEP_MONTH','ORG_AIRPRT','DES_AIRPRT','OD_PAX']
        if not set(cols).issubset(header):continue
        ix=[header.index(c) for c in cols]
        for n,row in enumerate(it,1):
            rows.append(tuple(row[i] for i in ix))
            if n%300000==0:print(f'Read {n:,} market records',flush=True)
    book.close();df=pd.DataFrame(rows,columns=['year','month','origin','destination','passengers']);del rows
    if df.empty or df.isna().any().any():raise ValueError('Empty or missing passenger input')
    if not np.isfinite(df.passengers).all() or (df.passengers<0).any():raise ValueError('Invalid passenger value')
    df['date']=pd.to_datetime(dict(year=df.year.astype(int),month=df.month.astype(int),day=1))
    if df.duplicated(['date','origin','destination']).any():raise ValueError('Duplicate market-month records: resolve upstream')
    df['market']=df.origin.astype(str)+'>'+df.destination.astype(str)
    panel=df.pivot(index='date',columns='market',values='passengers').sort_index()
    panel=panel.reindex(pd.date_range(panel.index.min(),panel.index.max(),freq='MS'))
    markets=df[['market','origin','destination']].drop_duplicates().set_index('market').reindex(panel.columns)
    book=load_workbook(airports,read_only=True,data_only=True);lookup=[]
    needed=['AIRPRT_CD','AIRPRT_NM','STATE_PROVNC_CD','CNTRY_CD','LAT_DEGREE_QTY','LAT_MINUTE_QTY','LAT_HEMSPHR_CD','LNGTD_DEGREE_QTY','LNGTD_MINUTE_QTY','LNGTD_HEMSPHR_CD']
    for sheet in book:
        it=iter(sheet.values);header=next(it)
        if 'AIRPRT_CD' not in header:continue
        ix={c:header.index(c) for c in needed}
        for row in it:lookup.append({c:row[i] for c,i in ix.items()})
    book.close();ap=pd.DataFrame(lookup)
    if ap.AIRPRT_CD.duplicated().any():raise ValueError('Airport lookup contains duplicate codes')
    for prefix,name in [('LAT','latitude'),('LNGTD','longitude')]:
        degree=pd.to_numeric(ap[prefix+'_DEGREE_QTY'],errors='coerce')
        minute=pd.to_numeric(ap[prefix+'_MINUTE_QTY'],errors='coerce').fillna(0)
        ap[name]=(degree+minute/60)*np.where(ap[prefix+'_HEMSPHR_CD'].isin(['S','W']),-1,1)
    ap=ap.set_index('AIRPRT_CD')
    for endpoint in ['origin','destination']:
        for col,key in [('AIRPRT_NM','name'),('STATE_PROVNC_CD','state'),('CNTRY_CD','country'),('latitude','latitude'),('longitude','longitude')]:
            markets[endpoint+'_'+key]=markets[endpoint].map(ap[col])
        markets[endpoint+'_state_group']=markets[endpoint+'_country'].fillna('UNKNOWN')+':'+markets[endpoint+'_state'].fillna('UNKNOWN')
    markets['domestic']=(markets.origin_country==markets.destination_country).astype(float)
    lat1=np.deg2rad(markets.origin_latitude);lat2=np.deg2rad(markets.destination_latitude)
    dlat=lat2-lat1;dlon=np.deg2rad(markets.destination_longitude-markets.origin_longitude)
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    markets['distance_km']=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    np.savez_compressed(out/'panel.npz',values=panel.to_numpy(dtype=float),dates=panel.index.strftime('%Y-%m-%d').to_numpy(dtype=str),markets=panel.columns.to_numpy(dtype=str))
    markets.reset_index().to_csv(out/'markets.csv',index=False)
    monthly=panel.sum(axis=1,min_count=1).to_frame('passengers');monthly['market_rows']=panel.notna().sum(axis=1)
    monthly.to_csv(out/'monthly_actuals.csv',index_label='date')
    audit={'rows':len(df),'markets':panel.shape[1],'calendar_months':len(panel),'first':str(panel.index.min().date()),'last':str(panel.index.max().date()),'duplicate_route_months':0,'unknown_airport_rows':int(markets[['origin_name','destination_name']].isna().any(axis=1).sum()),'missing_origin_state':int(markets.origin_state.isna().sum()),'missing_destination_state':int(markets.destination_state.isna().sum()),'sources':{Path(pax).name:fingerprint(pax),Path(airports).name:fingerprint(airports)},'geography_note':'Coordinates use degrees and minutes only; duplicated latitude-minute header prevents reliable seconds mapping. Approximate distance is sufficient for an exploratory predictor.'}
    (out/'input_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pax',required=True);p.add_argument('--airports',required=True);p.add_argument('--output',default='data/processed');a=p.parse_args();prepare(a.pax,a.airports,a.output)
