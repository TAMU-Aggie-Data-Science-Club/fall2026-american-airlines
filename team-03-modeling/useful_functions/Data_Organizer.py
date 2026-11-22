"""Team 3 route dictionaries using integrated ingestion; imports perform no I/O."""
import argparse
import json
from pathlib import Path
from aaforecast.data import load_pax, load_airports

US_CA = {'US', 'CA', 'PR', 'VI', 'GU', 'AS', 'MP'}


def build_dictionaries(frame, airports):
    region = airports['country'].isin(US_CA)
    airport_regions = {'USCA': airports.index[region].tolist(), 'NOTUSCA': airports.index[~region].tolist()}
    routes = {}
    for row in frame.itertuples(index=False):
        route = f'{row.org}->{row.des}'
        entry = routes.setdefault(route, {}).setdefault(int(row.month.year), {}).setdefault(
            int(row.month.month), {'OD_PAX': 0., 'OD_REV': 0.})
        entry['OD_PAX'] += float(row.od_pax)
        entry['OD_REV'] += float(row.od_rev_usd)
    return airport_regions, routes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('data/interim/dictionaries'))
    args = parser.parse_args(argv)
    airport_regions, routes = build_dictionaries(load_pax(use_cache=False), load_airports())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in (('airport_regions', airport_regions), ('routes', routes)):
        (args.out / f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
    print('Local dictionary outputs written. Keep them out of Git.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
