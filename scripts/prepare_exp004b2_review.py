"""Prepare read-only source comparisons for EXP004B-2; no scene decisions or split edits."""
import csv,hashlib,html,json,os
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=ROOT/'configs/splits/u3d_train_dev_v2.json'
 protected=[source,*(ROOT/'configs/splits').glob('u3d_*_v2.txt'),ROOT/'docs/U3D_TRAIN_DEV_V2.md',ROOT/'experiments/exp004b_dev_review/priority_review.csv',ROOT/'experiments/exp004b_dev_review/candidate_decisions.csv']
 before={p:sha(p) for p in protected}
 data=json.loads(source.read_text(encoding='utf-8'));relations=data['unreviewed_dev_train_retrieval_relations'];assert len(relations)==42
 dest=ROOT/'experiments/exp004b2_dev_train_review';assert not dest.exists(),'Refuse to overwrite review work'
 unique={}
 for row in relations:
  a,b=row['candidate'],row['related_train'];assert a in data['dev'] and b in data['train']
  key=tuple(sorted((a,b)))
  if key in unique:assert unique[key]==row,'Conflicting duplicate source records'
  else:unique[key]=row
 ordered=sorted(unique.values(),key=lambda x:(x['candidate_id'],x['related_train']))
 paths={ROOT/x[k] for x in ordered for k in ['candidate','related_train']}
 assert all(p.is_file() and p.resolve().is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve()) for p in paths)
 image_hashes={p:sha(p) for p in paths}
 dest.mkdir();out=dest/'outputs';out.mkdir()
 fields=['pair_id','dev_image','train_image','dev_candidate_id','similarity/source','phash63_hamming','gray32_ncc','decision','note','comparison_image']
 rows=[];sections=[];counts=Counter()
 for i,x in enumerate(ordered,1):
  pid=f'P{i:03d}';counts[x['candidate_id']]+=1
  canvas=Image.new('RGB',(1600,510),'#20242a');draw=ImageDraw.Draw(canvas)
  for offset,key,title in [(0,'candidate','DEV'),(800,'related_train','PROPOSED TRAIN')]:
   with Image.open(ROOT/x[key]) as im:
    rgb=im.convert('RGB');rgb.thumbnail((790,445),Image.Resampling.LANCZOS)
    canvas.paste(rgb,(offset+(800-rgb.width)//2,45+(445-rgb.height)//2))
   draw.text((offset+12,14),f'{pid} {title}: {Path(x[key]).name} | {x["candidate_id"]}',fill='white')
  draw.text((12,492),f'Original RGB thumbnails; no exposure adjustment. pHash distance={x["phash63_hamming"]}; NCC={float(x["gray32_ncc"]):.4f}',fill='white')
  name=f'{pid}_{x["candidate_id"]}_{Path(x["candidate"]).stem}_{Path(x["related_train"]).stem}.png'
  canvas.save(out/name)
  rows.append(dict(zip(fields,[pid,x['candidate'],x['related_train'],x['candidate_id'],x['basis'],x['phash63_hamming'],x['gray32_ncc'],'pending_user','',f'outputs/{name}'])))
  links=[os.path.relpath(ROOT/x[k],out).replace('\\','/') for k in ['candidate','related_train']]
  sections.append(f'<section id="{pid}" data-candidate="{x["candidate_id"]}"><h2>{pid} / {x["candidate_id"]}: {Path(x["candidate"]).name} vs {Path(x["related_train"]).name}</h2><p>Source: {html.escape(x["basis"])} | pHash distance: {x["phash63_hamming"]} | NCC: {x["gray32_ncc"]} | pending_user</p><a href="{name}" target="_blank"><img loading="lazy" src="{name}" alt="{pid} side-by-side comparison"></a><p><a href="{links[0]}" target="_blank">Open original DEV</a> | <a href="{links[1]}" target="_blank">Open original TRAIN</a></p></section>')
 with (dest/'review_dev_train.csv').open('x',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>NightUAV EXP004B-2</title><style>body{font:16px system-ui;margin:24px;background:#111820;color:#e5e7eb}a{color:#8ed5ff}section{margin:32px 0;border-top:1px solid #52606c}img{width:100%;max-width:1600px}header{position:sticky;top:0;background:#111820;padding:12px}select{padding:6px}</style><header><h1>EXP004B-2: dev to train manual review</h1><p>Fill decision and note in <a href="../review_dev_train.csv">review_dev_train.csv</a>. same = same real scene; different = different real scenes; uncertain = cannot reliably decide. Do not decide from similarity scores alone.</p><p>This page is a static contact sheet; it does not save decisions or update from CSV. Pair IDs map to CSV rows. Original files open locally. Images preserve original brightness; use full-size originals when details are unclear.</p><label>Candidate: <select id="filter"><option value="all">All</option>'''+''.join(f'<option value="{k}">{k} ({v})</option>' for k,v in sorted(counts.items()))+'''</select></label></header>'''+''.join(sections)+'''<script>document.querySelector('#filter').onchange=function(){document.querySelectorAll('section').forEach(s=>s.hidden=this.value!=='all'&&s.dataset.candidate!==this.value)}</script></html>'''
 (out/'index.html').write_text(page,encoding='utf-8')
 with (dest/'review_dev_train.csv').open(encoding='utf-8-sig',newline='') as f:loaded=list(csv.DictReader(f))
 assert len(loaded)==len(ordered)==42 and all(x['decision']=='pending_user' for x in loaded)
 assert len({tuple(sorted((x['dev_image'],x['train_image']))) for x in loaded})==42
 for row in loaded:
  with Image.open(dest/row['comparison_image']) as im:assert im.size==(1600,510) and im.mode=='RGB';im.verify()
 assert all(sha(p)==h for p,h in before.items()),'Protected source changed'
 assert all(sha(p)==h for p,h in image_hashes.items()),'Original image changed'
 manifest={'source':source.relative_to(ROOT).as_posix(),'source_sha256':before[source],'source_relations':42,'unique_pairs':len(rows),'duplicate_pairs_removed':len(relations)-len(rows),'by_candidate':dict(sorted(counts.items())),'comparisons_verified':len(rows),'all_decisions_pending':True,'protected_sources_unchanged':True,'original_images_unchanged':True,'source_image_sha256':{p.relative_to(ROOT).as_posix():h for p,h in sorted(image_hashes.items())},'script_sha256':sha(Path(__file__))}
 (dest/'generation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 (dest/'README.md').write_text('# EXP004B-2 manual review\n\n42 unique dev-to-train pairs extracted from the v2 candidate JSON; no new retrieval or automatic same/different decisions.\n\nOpen [contact sheet](outputs/index.html), optionally filter by candidate. Pair IDs correspond exactly to [review CSV](review_dev_train.csv). Fill decision with `same`, `different`, or `uncertain`; record your reasoning in note. Preserve paths and pair IDs. The HTML is static and does not save edits.\n\nComparisons use RGB thumbnails with preserved aspect ratio and unchanged exposure; click original links for detail. Similarity metrics are retrieval evidence, not scene labels. Outputs are ignored by Git. No v2 changes, training, commit or push.\n\nGenerator: `python -B scripts/prepare_exp004b2_review.py`; refuses to overwrite an existing review directory. Source hashes, per-candidate counts and verification results: [manifest](generation_manifest.json).\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in manifest.items() if k!='source_image_sha256'},indent=2))
if __name__=='__main__':main()
