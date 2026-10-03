"""Read-only EXP004B progress, link and transitive-conflict checks; never publishes splits."""
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image

def inspect(folder):
    hashes={};tables={};errors=[];parent={};decisions=defaultdict(set)
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    for name in ['candidate_decisions.csv','association_decisions.csv','priority_review.csv']:
        p=folder/name;hashes[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        tables[name]=list(csv.DictReader(p.open(encoding='utf-8-sig')))
        allowed={'','pending_user','keep_for_further_review','drop','defer'} if name.startswith('candidate_') else {'','pending_user','same_scene','different_scene','uncertain'}
        seen=set()
        for row in tables[name]:
            key=row['candidate_id'] if name.startswith('candidate_') else tuple(sorted([row['path_a'],row['path_b']]))
            if key in seen:errors.append(f'duplicate row: {name}: {key}')
            seen.add(key)
            d=row['decision'].strip()
            if d not in allowed:errors.append(f'unknown decision: {name}: {key}: {d}')
            if not name.startswith('candidate_') and d not in {'','pending_user'}:decisions[key].add(d)
    full={tuple(sorted([r['path_a'],r['path_b']])) for r in tables['association_decisions.csv']}
    for r in tables['priority_review.csv']:
        if tuple(sorted([r['path_a'],r['path_b']])) not in full:errors.append('priority pair absent from full review: '+str(r))
    for pair,ds in decisions.items():
        if len(ds)>1:errors.append(f'conflicting tables: {pair}: {sorted(ds)}')
        if ds=={'same_scene'}:parent[find(pair[0])]=find(pair[1])
    for (a,b),ds in decisions.items():
        if 'different_scene' in ds and find(a)==find(b):errors.append(f'transitive same/different conflict: {a}, {b}')
    html=folder/'outputs/index.html';links=re.findall(r'<img\s+src="([^"]+)"',html.read_text(encoding='utf-8'))
    for link in links:
        p=(html.parent/link).resolve()
        if not p.is_relative_to(html.parent.resolve()):errors.append('unsafe image link: '+link);continue
        try:
            with Image.open(p) as im:im.verify()
        except Exception as e:errors.append(f'broken image: {link}: {e}')
    assert hashes=={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in hashes},'Review changed while reading; rerun'
    return {'status':'review_only_NOT_APPROVED','training_allowed':False,'counts':{n:dict(Counter(r['decision'].strip() or 'blank' for r in rows)) for n,rows in tables.items()},'contact_images_checked':len(links),'errors':errors,'review_sha256':hashes,'limitations':'Pending/uncertain decisions do not prove independence. No changes to review CSVs or splits.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,default=Path(__file__).resolve().parents[1]/'experiments/exp004b_dev_review');ap.add_argument('--output',type=Path)
    a=ap.parse_args();report=inspect(a.folder)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if report['errors']:raise SystemExit(1)
if __name__=='__main__':main()
