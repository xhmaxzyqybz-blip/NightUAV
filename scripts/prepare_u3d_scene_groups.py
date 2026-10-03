"""Reuse EXP003 hashes; retrieve all train/train pairs and conservatively group risks."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];E=R/'experiments/exp004_di_retinex_baseline';O=E/'outputs';A=E/'audit';O.mkdir(parents=True,exist_ok=True);A.mkdir(exist_ok=True)
if (A/'retrieval_summary.json').exists():raise FileExistsError('Completed audit exists; overwrite disabled')
def dump(name,x):
 with (A/name).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
inv=json.loads((R/'experiments/exp003_di_retinex_smoke/audit/inventory.json').read_text(encoding='utf-8'));train=[r for r in inv if r['split']=='train'];look={r['source']:i for i,r in enumerate(inv)}
# Snapshot hashes of pre-existing reports/scripts for final preservation check.
old={}
for folder in ['docs','scripts','experiments/exp001_di_retinex','experiments/exp002_baseline_diagnosis','experiments/exp003_di_retinex_smoke']:
 for p in (R/folder).rglob('*'):
  if p.is_file() and 'outputs' not in p.parts and p.name!=Path(__file__).name:old[p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
if not (A/'previous_artifacts_sha256.json').exists():dump('previous_artifacts_sha256.json',old)
thumbs=[];features=[]
for idx,row in enumerate(train):
 p=R/row['source']
 if not p.is_file():raise RuntimeError('Missing source '+str(p))
 with Image.open(p) as im:
  im.draft('RGB',(320,180));im=im.convert('RGB');thumbs.append(im.resize((240,135),Image.Resampling.LANCZOS));g=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float).flatten()
 g-=g.mean();features.append(g/max(np.linalg.norm(g),1e-12))
 if (idx+1)%200==0:print('train thumbnails',idx+1,flush=True)
B=np.array([[x=='1' for x in row['phash63']] for row in train]);dist=(B[:,None,:]!=B[None,:,:]).sum(2);F=np.array(features);ncc=F@F.T
edges=[]
for i,j in np.argwhere(np.triu((dist<=14)|(ncc>=.90),1)):
 edges.append({'a':train[i]['source'],'b':train[j]['source'],'hamming':int(dist[i,j]),'ncc':float(ncc[i,j]),'reason':'train_all_pairs_threshold','status':'pending_visual_review'})
# Weak top-2 neighbors retained as review queue, not automatic same-scene assignments.
nearest=[]
for i in range(800):
 score=ncc[i].copy();score[i]=-2
 for j in np.argsort(-score)[:2]:nearest.append({'a':train[i]['source'],'b':train[j]['source'],'hamming':int(dist[i,j]),'ncc':float(ncc[i,j])})
confirmed_parent=list(range(len(inv)));risk_parent=list(range(len(inv)))
def find(p,i):
 while p[i]!=i:p[i]=p[p[i]];i=p[i]
 return i
def union(p,i,j):
 i=find(p,i);j=find(p,j)
 if i!=j:p[max(i,j)]=min(i,j)
oldrev=json.loads((R/'experiments/exp003_di_retinex_smoke/audit/visual_review.json').read_text(encoding='utf-8'))[:27];oldpairs=json.loads((R/'experiments/exp003_di_retinex_smoke/audit/visualized_pairs.json').read_text(encoding='utf-8'))[:27]
relations=[]
for rev in oldrev:
 i=look[rev['train']];j=look[rev['val']];status=rev['classification'];relations.append({'a':rev['train'],'b':rev['val'],'basis':'EXP003 visual layout review','status':status})
 union(risk_parent,i,j)
 if status in ['exact_bytes_confirmed','same_scene_visual_strong']:union(confirmed_parent,i,j)
hashes={}
for i,row in enumerate(inv):hashes.setdefault(row['sha256'],[]).append(i)
for group in hashes.values():
 for i in group[1:]:
  union(confirmed_parent,group[0],i);union(risk_parent,group[0],i);relations.append({'a':inv[group[0]]['source'],'b':inv[i]['source'],'basis':'EXP003 full SHA256 equality','status':'exact_bytes_confirmed'})
for edge in edges:union(risk_parent,look[edge['a']],look[edge['b']])
groups={}
for i,row in enumerate(train):groups.setdefault(find(risk_parent,look[row['source']]),[]).append(i)
ordered=sorted(groups.values(),key=lambda z:(-len(z),train[z[0]]['source']))
for gn,group in enumerate(ordered):
 # All train thumbnails are available for manual completion, preserving filename labels.
 for page in range((len(group)+23)//24):
  subset=group[page*24:page*24+24];canvas=Image.new('RGB',(960,6*160),'white');draw=ImageDraw.Draw(canvas)
  for q,i in enumerate(subset):
   x=q%4*240;y=q//4*160;canvas.paste(thumbs[i],(x,y+25));draw.text((x,y),f'G{gn:03d} {Path(train[i]["source"]).name}',fill='black')
  with (O/f'group_{gn:03d}_{page:02d}.png').open('xb') as f:canvas.save(f,format='PNG')
# Render all strong candidate pairs, so pending rows always have an actual comparison image.
for q,edge in enumerate(edges):
 i=look[edge['a']];j=look[edge['b']];canvas=Image.new('RGB',(480,165),'white');draw=ImageDraw.Draw(canvas);canvas.paste(thumbs[i],(0,30));canvas.paste(thumbs[j],(240,30));draw.text((2,2),f'{q:04d} {Path(edge["a"]).stem}/{Path(edge["b"]).stem} d={edge["hamming"]} ncc={edge["ncc"]:.3f}',fill='black')
 with (O/f'pair_{q:04d}.png').open('xb') as f:canvas.save(f,format='PNG')
 edge['comparison_image']=f'experiments/exp004_di_retinex_baseline/outputs/pair_{q:04d}.png'
gid={train[i]['source']:f'RISK_{gn:03d}' for gn,g in enumerate(ordered) for i in g};records=[]
for row in inv:
 idx=look[row['source']];records.append({**row,'scene_group_id':f'SCENE_PROVISIONAL_{find(confirmed_parent,idx):04d}','conservative_group_id':gid.get(row['source'],'ORIGINAL_VAL_EXCLUDED'),'grouping_basis':'transitive exact-byte / prior strong visual edges; new similarity edges only form conservative risk blocks','confirmation_status':'provisional_not_sequence_verified','original_split':row['split'],'usage':'pending' if row['split']=='train' else 'original_val_excluded'})
dump('scene_records_draft.json',records);dump('confirmed_or_visual_relations.json',relations);dump('train_similarity_candidates.json',edges);dump('train_top2_neighbors.json',nearest)
dump('retrieval_summary.json',{'train_images':800,'train_train_pairs':319600,'strong_candidates':len(edges),'top2_rows':len(nearest),'risk_groups':len(ordered),'group_sizes':[len(x) for x in ordered],'threshold':'Hamming<=14 or NCC>=0.90; retrieval-only, not scene confirmation','old_inventory_sha256':hashlib.sha256((R/'experiments/exp003_di_retinex_smoke/audit/inventory.json').read_bytes()).hexdigest()})
print('SUMMARY',len(edges),len(ordered),[len(x) for x in ordered[:30]],flush=True)
