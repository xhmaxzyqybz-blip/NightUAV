"""Freeze blocked v1 manifests; scene confirmation and conservative risk blocks remain distinct."""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[1];E=R/'experiments/exp004_di_retinex_baseline';A=E/'audit';S=R/'configs/splits';S.mkdir(parents=True,exist_ok=True)
rows=json.loads((A/'scene_records_draft.json').read_text(encoding='utf-8'));edges=json.loads((A/'confirmed_or_visual_relations.json').read_text(encoding='utf-8'));look={r['source']:i for i,r in enumerate(rows)};parent=list(range(len(rows)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):
 i=find(i);j=find(j)
 if i!=j:parent[max(i,j)]=min(i,j)
for edge in edges:
 if edge['status'] in ['exact_bytes_confirmed','same_scene_visual_strong']:union(look[edge['a']],look[edge['b']])
# Reviewed contact sheets: explicit recognizable-layout associations, not hash-only labels.
visual_sets={
'bright_roof_office_complex':['0032','0046','0225','0261','0344','0532','0635','0676','0686','0728','0770','0796','0199','0704','0716'],
'avenue_stadium_oblique':['0041','0071','0091','0287','0416','0461','0575','0619','0625','0674','0784','0787'],
'stadium_front':['0076','0233','0280','0396','0437','0451','0695','0717','0763'],
'commercial_curved_tower':['0144','0162','0190','0265'],
'road_between_highrises':['0166','0210','0223','0546','0594'],
'commercial_strip':['0022','0694','0110'],
'road_dark_building':['0153','0629'],
'commercial_lowrise':['0078','0282'],
}
labels={};newedges=[]
for category,ids in visual_sets.items():
 paths=[f'datasets/U3D/lowlight/train/{x}.jpg' for x in ids]
 for p in paths:labels[p]=category
 for p in paths[1:]:union(look[paths[0]],look[p]);newedges.append({'a':paths[0],'b':p,'basis':'EXP004 visual distinctive layout; group_000..005 contact sheets','status':'visual_same_scene','category':category})
# Preserve conservative transitive risk grouping, and merge blocks bridged by reviewed relations.
risknames=sorted(set(r['conservative_group_id'] for r in rows if r['original_split']=='train'));rp={x:x for x in risknames}
def rf(x):
 while rp[x]!=x:rp[x]=rp[rp[x]];x=rp[x]
 return x
def ru(x,y):
 x=rf(x);y=rf(y)
 if x!=y:rp[max(x,y)]=min(x,y)
for group in visual_sets.values():
 ns=[rows[look[f'datasets/U3D/lowlight/train/{n}.jpg']]['conservative_group_id'] for n in group]
 for n in ns[1:]:ru(ns[0],n)
# Different views around a stadium are conservatively kept in one exclusion block, not asserted exact scene identity.
ru('RISK_001','RISK_002')
scene_evidence=collections.defaultdict(list)
for edge in edges+newedges:
 if edge['status'] in ['exact_bytes_confirmed','same_scene_visual_strong','visual_same_scene']:
  for p in [edge['a'],edge['b']]:scene_evidence[p].append(edge)
for i,row in enumerate(rows):
 row['scene_group_id']=f'SCENE_{find(i):04d}'
 row['scene_category']=labels.get(row['source'],'unclassified')
 row['grouping_basis']=scene_evidence[row['source']] or [{'basis':'no confirmed scene association; singleton ID is NOT proof of independence'}]
 row['confirmation_status']='visual_association_without_sequence_metadata' if any(x['status']!='exact_bytes_confirmed' for x in scene_evidence[row['source']]) else 'exact_duplicate_association_only' if scene_evidence[row['source']] else 'unreviewed_scene'
 if row['original_split']=='train':
  row['conservative_group_id']=rf(row['conservative_group_id']);row['usage']='quarantine';row['quarantine_reason']='scene partition not released; unresolved false merges and missed links; no train/dev admission'
 else:row['usage']='original_val_excluded'
# All 800 are conservatively withheld until release; do not invent an executable split.
for split in ['train','dev','quarantine']:
 with (S/f'u3d_scene_{split}_v1.txt').open('x',encoding='utf-8') as f:
  f.write('# EXP004 v1 BLOCKED: not an approved training split. Project-relative paths.\n')
  if split=='quarantine':f.write('\n'.join(sorted(r['source'] for r in rows if r['original_split']=='train'))+'\n')
with (S/'u3d_scene_groups_v1.json').open('x',encoding='utf-8') as f:json.dump({'version':'v1','status':'blocked_pending_scene_review','training_allowed':False,'original_val_policy':'excluded from training/model selection','test_policy':'not read or used','records':rows},f,ensure_ascii=False,indent=2)
with (A/'exp004_visual_associations.json').open('x',encoding='utf-8') as f:json.dump(newedges,f,ensure_ascii=False,indent=2)
meta={'version':'v1','status':'blocked_pending_scene_review','training_allowed':False,'blocking_reasons':['Automatic risk blocks contain different scenes','Visually same office complex spans prior blocks 000/004/005','Most original-train scenes remain unreviewed; singleton IDs cannot prove independence'],'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('u3d_scene_*v1.*')}}
with (S/'u3d_scene_v1_metadata.json').open('x',encoding='utf-8') as f:json.dump(meta,f,ensure_ascii=False,indent=2)
print('Quarantine',sum(r['usage']=='quarantine' for r in rows),'risk groups',len(set(r['conservative_group_id'] for r in rows if r['original_split']=='train')),'scene groups',len(set(r['scene_group_id'] for r in rows if r['original_split']=='train')))
