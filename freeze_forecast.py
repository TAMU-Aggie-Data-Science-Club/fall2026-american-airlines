"""Keep an immutable, content-addressed issue for future prospective scoring."""
from pathlib import Path
import hashlib,json,zipfile,shutil

def digest(data):return hashlib.sha256(data).hexdigest()

def freeze(root):
    root=Path(root);results=root/'data/processed/results';cfg=json.loads((root/'config.json').read_text());summary=json.loads((results/'summary.json').read_text())
    names=['model.py','selection.py','market_methods.py','config.json'];hashes={n:digest((root/n).read_bytes()) for n in names}
    forecast=(results/'market_year2_monthly.csv').read_bytes()
    manifest={'forecast_origin':summary['origin'],'issued_as_of':cfg['as_of'],'forecast_start':summary['forecast_start'],'forecast_end':summary['forecast_end'],'selected_model':summary['selected_model'],
              'source_hashes':hashes,'panel_sha256':digest((root/'data/processed/panel.npz').read_bytes()),'forecast_sha256':digest(forecast),'status':'Issued forecast; outcomes not yet known'}
    issue_id=digest(json.dumps(manifest,sort_keys=True).encode())[:16];manifest['issue_id']=issue_id
    folder=root/'data/processed/issued';folder.mkdir(exist_ok=True)
    target=folder/(summary['origin']+'_'+issue_id+'.zip')
    if not target.exists():
        with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json',json.dumps(manifest,indent=2));z.writestr('market_forecast.csv',forecast)
            z.write(results/'total_year2_monthly.csv','total_forecast.csv')
            for n in names:z.write(root/n,n)
    with zipfile.ZipFile(target) as z:
        if digest(z.read('market_forecast.csv'))!=manifest['forecast_sha256']:raise ValueError('Forecast issue integrity failure')
    shutil.copyfile(target,results/'forecast_issue.zip')
    (results/'forecast_issue_record.json').write_text(json.dumps({**manifest,'archive':str(target.relative_to(root))},indent=2))
    print('Frozen forecast:',target)
    return target

if __name__=='__main__':freeze(Path(__file__).parent)
