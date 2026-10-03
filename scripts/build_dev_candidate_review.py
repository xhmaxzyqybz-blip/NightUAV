"""Candidate development review only; NEVER publishes an approved train/dev split."""
from pathlib import Path
import json,csv,hashlib,math,collections
import numpy as np
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1];E=R/'experiments/exp004b_dev_review';O=E/'outputs';O.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads((R/p).read_text(encoding='utf-8'))
def dump(name,v):
 with (E/name).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
paths=['configs/splits/u3d_scene_groups_v1.json','experiments/exp004_di_retinex_baseline/audit/confirmed_or_visual_relations.json','experiments/exp004_di_retinex_baseline/audit/exp004_visual_associations.json','experiments/exp004_di_retinex_baseline/audit/train_similarity_candidates.json','experiments/exp004_di_retinex_baseline/audit/train_top2_neighbors.json','experiments/exp003_di_retinex_smoke/audit/visual_review.json','experiments/exp003_di_retinex_smoke/audit/exact_duplicates.json','experiments/exp002_baseline_diagnosis/diagnostic_manifest.json']
loaded={p:read(p) for p in paths};review_csv=R/'experiments/exp004_di_retinex_baseline/audit/review_queue.csv'
with review_csv.open(encoding='utf-8',newline='') as f:previous_queue=list(csv.DictReader(f))
dump('source_reviews.json',{'sources':{p:hashlib.sha256((R/p).read_bytes()).hexdigest() for p in paths},'review_queue_sha256':hashlib.sha256(review_csv.read_bytes()).hexdigest(),'review_queue_rows':len(previous_queue),'top2_rows':len(loaded[paths[4]]),'status':'prior assistant visual associations, NOT user final approval'})
allrows=loaded[paths[0]]['records'];rows=[x for x in allrows if x['original_split']=='train'];lookup={x['source']:i for i,x in enumerate(rows)}
excluded={x['source'] for x in loaded[paths[-1]]['images']};duplicates={p for g in loaded[paths[-2]] for p in g};categories=collections.defaultdict(list)
for row in rows:
 if row['scene_category']!='unclassified' and row['source'] not in excluded|duplicates:categories[row['scene_category']].append(row['source'])
# Extra locations explicitly reviewed in EXP003 paired sheets, not new hash-inferred semantic labels.
for cat,ids in {'skyline_courtyard':['0009','0131','0216'],'park_highrise_road':['0114','0127','0343'],'commercial_night_market':['0551','0758','0646'],'near_mall_street':['0792','0179'],'residential_road':['0049']}.items():
 for n in ids:
  p=f'datasets/U3D/lowlight/train/{n}.jpg'
  if p not in excluded|duplicates:categories[cat].append(p)
selected=[]
# Round-robin diversity across known visual categories, up to five per category.
for k in range(5):
 for cat in sorted(categories):
  v=sorted(categories[cat])
  if k<len(v) and len(selected)<40:selected.append((v[k],cat))
assert 30<=len(selected)<=50 and len(set(p for p,c in selected))==len(selected)
thumb=[];F=[];lighting={}
for i,row in enumerate(rows):
 with Image.open(R/row['source']) as im:
  im.draft('RGB',(320,180));im=im.convert('RGB');thumb.append(im.resize((240,135),Image.Resampling.LANCZOS));g=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float).flatten()
 lighting[row['source']]={'mean_gray':float(g.mean()/255),'std_gray':float(g.std()/255),'dark_fraction':float((g<25.5).mean()),'bright_fraction':float((g>229.5).mean())}
 g-=g.mean();F.append(g/max(np.linalg.norm(g),1e-12))
 if (i+1)%200==0:print('retrieval thumbnails',i+1,flush=True)
F=np.array(F);B=np.array([[c=='1' for c in r['phash63']] for r in rows]);selidx=[lookup[p] for p,cat in selected];NCC=F[selidx]@F.T;HD=(B[selidx,None,:]!=B[None,:,:]).sum(2)
# Full scores are local visualization/retrieval artifacts, excluded from Git.
with (O/'all_candidate_train_scores.csv').open('x',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['candidate','other_original_train','phash63_hamming','gray32_ncc'])
 for k,(p,cat) in enumerate(selected):
  for j,row in enumerate(rows):
   if j!=selidx[k]:w.writerow([p,row['source'],int(HD[k,j]),float(NCC[k,j])])
relations=[];candidate_records=[];all_exclusions={p for p,cat in selected};priority_pairs=set();all_pairs=set();html=['<html><meta charset="utf-8"><title>EXP004B review</title><body><h1>Candidate review only / pending user decisions</h1><p>No train/dev v2 has been published.</p>']
for k,(p,cat) in enumerate(selected):
 i=lookup[p];row=rows[i];topn=[int(x) for x in np.argsort(-NCC[k]) if x!=i][:6];toph=[int(x) for x in np.argsort(HD[k],kind='stable') if x!=i][:6]
 known={j for j,r in enumerate(rows) if j!=i and (r['scene_group_id']==row['scene_group_id'] or r['conservative_group_id']==row['conservative_group_id'] or r['sha256']==row['sha256'])}
 strong={j for j in range(800) if j!=i and (HD[k,j]<=14 or NCC[k,j]>=.90)}
 related=set(topn+toph)|known|strong
 # Include pending/confirmed prior edges incident on candidate even if current threshold misses them.
 for edge in loaded[paths[1]]+loaded[paths[2]]+loaded[paths[3]]:
  aa=edge['a'];bb=edge['b']
  if aa==p and bb in lookup:related.add(lookup[bb])
  if bb==p and aa in lookup:related.add(lookup[aa])
 pending_risk={j for j in known|strong if rows[j]['scene_group_id']!=row['scene_group_id']}
 recommendation='suggest_drop_or_defer_many_uncertain_links' if len(pending_risk)>=8 else 'review_required'
 ordered=sorted(related,key=lambda j:(0 if j in pending_risk else 1 if j in known else 2,-float(NCC[k,j]),int(HD[k,j])))
 priority=set(ordered[:4])|pending_risk
 candidate_records.append({'candidate_id':f'D{k+1:02d}','source':p,'original_split':'train','sha256':row['sha256'],'scene_group_id':row['scene_group_id'],'conservative_group_id':row['conservative_group_id'],'scene_category':cat,'category_basis':'previous assistant visual review; user approval pending','lighting_thumbnail':lighting[p],'related_images':len(related),'uncertain_known_or_strong_links':len(pending_risk),'priority_relations':len(priority),'recommendation':recommendation,'review_status':'pending_user','not_exp002_diagnostic':True,'smoke_use_disqualifies_future_independent_training':False})
 for j in ordered:
  q=rows[j]['source'];basis=[]
  if row['sha256']==rows[j]['sha256']:basis.append('exact_sha256')
  if row['scene_group_id']==rows[j]['scene_group_id']:basis.append('existing_visual_or_exact_transitive_component')
  if row['conservative_group_id']==rows[j]['conservative_group_id']:basis.append('existing_conservative_risk_block_not_confirmed_scene')
  if j in topn:basis.append('top6_ncc')
  if j in toph:basis.append('top6_phash')
  if j in strong:basis.append('phash<=14_or_ncc>=0.90')
  pair=tuple(sorted([p,q]));all_pairs.add(pair)
  if j in priority:priority_pairs.add(pair)
  relations.append({'candidate_id':f'D{k+1:02d}','candidate':p,'related_train':q,'phash63_hamming':int(HD[k,j]),'gray32_ncc':float(NCC[k,j]),'basis':';'.join(basis),'priority':j in priority,'decision':'pending_user','reviewer_note':''});all_exclusions.add(q)
 pages=[]
 for page in range(math.ceil(len(ordered)/12)):
  subset=ordered[page*12:page*12+12];canvas=Image.new('RGB',(960,690),'white');d=ImageDraw.Draw(canvas);canvas.paste(thumb[i],(0,25));d.text((0,5),f'D{k+1:02d} candidate {Path(p).name}',fill='black');d.text((250,30),cat+' / pending user review',fill='black');d.text((250,55),recommendation,fill='black')
  for z,j in enumerate(subset):
   x=(z%4)*240;y=180+(z//4)*170;canvas.paste(thumb[j],(x,y+30));d.text((x,y),f'{Path(rows[j]["source"]).stem} H{int(HD[k,j])} NCC{NCC[k,j]:.3f}'+(' PRIORITY' if j in priority else ''),fill='black')
  name=f'D{k+1:02d}_page{page+1:02d}.png'
  with (O/name).open('xb') as f:canvas.save(f,format='PNG')
  pages.append(name)
 html.append(f'<h2>D{k+1:02d} {p}</h2><p>{cat}: {recommendation}; related={len(related)}, priority={len(priority)}</p>'+''.join(f'<a href="{n}"><img src="{n}" width="960"></a>' for n in pages))
# Expand exclusions through ALL known scene and conservative blocks (including transitive edges).
change=True
while change:
 before=len(all_exclusions);sg={rows[lookup[p]]['scene_group_id'] for p in all_exclusions};rg={rows[lookup[p]]['conservative_group_id'] for p in all_exclusions};hs={rows[lookup[p]]['sha256'] for p in all_exclusions}
 all_exclusions|={r['source'] for r in rows if r['scene_group_id'] in sg or r['conservative_group_id'] in rg or r['sha256'] in hs};change=len(all_exclusions)>before
pool=[r['source'] for r in rows if r['source'] not in all_exclusions]
assert not ({rows[lookup[p]]['scene_group_id'] for p in pool}&{rows[lookup[p]]['scene_group_id'] for p in all_exclusions})
assert not ({rows[lookup[p]]['conservative_group_id'] for p in pool}&{rows[lookup[p]]['conservative_group_id'] for p in all_exclusions})
with (E/'candidate_associations.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=relations[0].keys());w.writeheader();w.writerows(relations)
with (O/'index.html').open('x',encoding='utf-8') as f:f.write(''.join(html)+'</body></html>')
with (E/'candidate_dev_manifest.json').open('x',encoding='utf-8') as f:json.dump({'status':'pending_user_review_NOT_APPROVED','training_allowed':False,'candidates':candidate_records},f,ensure_ascii=False,indent=2)
dump('review_only_pool.json',{'status':'NOT_A_TRAINING_SPLIT','training_allowed':False,'excluded_pending_review':sorted(all_exclusions),'remaining_original_train_for_review_only':pool,'warning':'No v2 release; unrecognized scene links may remain; future formal run MUST initialize randomly, never use EXP004A checkpoint'})
summary={'candidate_images':len(selected),'queries_against_other_train':len(selected)*799,'candidate_category_counts':dict(collections.Counter(x['scene_category'] for x in candidate_records)),'relations_rows':len(relations),'unique_relation_pairs':len(all_pairs),'priority_unique_pairs':len(priority_pairs),'related_unique_images':len({x['related_train'] for x in relations}),'suggest_drop_or_defer':sum(x['recommendation'].startswith('suggest') for x in candidate_records),'review_only_pool_size':len(pool),'excluded_closure_size':len(all_exclusions),'known_group_crossing_after_exclusion':False,'approved_v2':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};dump('summary.json',summary);print(json.dumps(summary),flush=True)
