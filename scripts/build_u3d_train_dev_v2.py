"""Build reviewed U3D v2 candidate manifests. No image writes, model imports or training."""
import csv,hashlib,json,collections,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/exp004b_dev_review'
KEEP={'D07','D08','D09','D10','D19','D28','D29','D31'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.reader(f))
 assert rows and len(set(rows[0]))==len(rows[0]),str(p)
 assert all(len(x)==len(rows[0]) for x in rows[1:]),'Malformed CSV: '+str(p)
 return [dict(zip(rows[0],x)) for x in rows[1:]]
def main():
 outputs=[ROOT/'configs/splits'/n for n in ['u3d_train_dev_v2.json','u3d_train_v2.txt','u3d_dev_v2.txt','u3d_quarantine_v2.txt']]+[ROOT/'docs/U3D_TRAIN_DEV_V2.md']
 assert not any(p.exists() for p in outputs),'Refuse to overwrite existing v2 artifacts'
 sources=[BASE/n for n in ['priority_review.csv','candidate_decisions.csv','association_decisions.csv','candidate_associations.csv','candidate_dev_manifest.json']]
 source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
 priority=read_csv(sources[0]);candidates=read_csv(sources[1]);full=read_csv(sources[2]);associations=read_csv(sources[3])
 assert len(priority)==60 and set(x['decision'] for x in priority)<={'same','different','uncertain'}
 assert collections.Counter(x['decision'] for x in candidates)=={'keep':8,'exclude':9,'defer':23}
 assert {x['candidate_id'] for x in candidates if x['decision']=='keep'}==KEEP
 assert len({x['candidate_id'] for x in candidates})==40 and len({x['source'] for x in candidates})==40
 metadata={x['candidate_id']:x for x in json.loads(sources[4].read_text(encoding='utf-8-sig'))['candidates']}
 for x in candidates:assert x['source']==metadata[x['candidate_id']]['source']
 universe=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'datasets/U3D/lowlight/train').iterdir() if p.is_file())
 assert len(universe)==800
 U=set(universe);parent={x:x for x in universe}
 def find(x):
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 def pair(x):return tuple(sorted((x['path_a'],x['path_b'])))
 assert len({pair(x) for x in priority})==60
 full_pairs={pair(x) for x in full};assert all(pair(x) in full_pairs for x in priority)
 # Full review is an old template here. Never interpret pending as different.
 assert all(x['decision']=='pending_user' for x in full),'Additional full-table decisions require explicit reconciliation'
 for x in priority:
  assert x['path_a'] in U and x['path_b'] in U and x['path_a']!=x['path_b']
  if x['decision']=='same':parent[find(x['path_a'])]=find(x['path_b'])
 conflicts=[x for x in priority if x['decision']=='different' and find(x['path_a'])==find(x['path_b'])]
 assert not conflicts,'STOP: same/different conflicts: '+json.dumps(conflicts)
 reps=sorted((x for x in candidates if x['decision']=='keep'),key=lambda x:x['candidate_id'])
 assert len({find(x['source']) for x in reps})==8,'STOP: multiple dev representatives in one same group'
 groups=[]
 for x in reps:
  members=sorted(p for p in universe if find(p)==find(x['source']))
  groups.append({'group_id':'MANUAL_'+x['candidate_id'],'candidate_id':x['candidate_id'],'representative':x['source'],'members':members,'size':len(members),'scene_category_descriptive_only':metadata[x['candidate_id']]['scene_category'],'same_evidence':[e for e in priority if e['decision']=='same' and e['path_a'] in members]})
 dev={x['source'] for x in reps};isolated={p for g in groups for p in g['members']};quarantine=isolated-dev;train=U-isolated
 assert train and dev
 assert not(train&dev or train&quarantine or dev&quarantine)
 assert train|dev|quarantine==U
 assert not({find(p) for p in train}&{find(p) for p in dev})
 hashes={p:sha(ROOT/p) for p in universe}
 assert all((ROOT/p).resolve().is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve()) for p in universe)
 assert not({hashes[p] for p in train}&{hashes[p] for p in dev}),'STOP: train/dev exact-byte duplicate'
 assert not({hashes[p] for p in train}&{hashes[p] for p in quarantine}),'STOP: train/quarantine exact-byte duplicate'
 assert len({hashes[p] for p in dev})==8,'STOP: duplicate dev contents'
 for x in candidates:assert hashes[x['source']]==metadata[x['candidate_id']]['sha256'],'Candidate changed'
 assert all(x['source'] in quarantine for x in candidates if x['decision']=='exclude'),'Excluded candidate not linked to a kept scene'
 reviewed={pair(x):x['decision'] for x in priority}
 pending_cross=[]
 for x in associations:
  a,b=x['candidate'],x['related_train'];key=tuple(sorted((a,b)))
  if a in dev and b in train and key not in reviewed:pending_cross.append(x)
 # Assistant visual hypotheses are retained as risks, never promoted to human decisions.
 old_visual=[]
 for n in ['confirmed_or_visual_relations.json','exp004_visual_associations.json']:
  p=ROOT/'experiments/exp004_di_retinex_baseline/audit'/n
  source_hashes[p.relative_to(ROOT).as_posix()]=sha(p)
  for x in json.loads(p.read_text(encoding='utf-8-sig')):
   if (x['a'] in isolated and x['b'] in train) or (x['b'] in isolated and x['a'] in train):old_visual.append(x)
 duplicates=collections.defaultdict(list)
 for p,h in hashes.items():duplicates[h].append(p)
 checks={'csv_column_counts_valid':True,'csv_format_repairs':[], 'same_different_conflicts':[], 'train_dev_path_intersection':0,'train_quarantine_path_intersection':0,'dev_quarantine_path_intersection':0,'train_dev_sha256_intersection':0,'train_quarantine_sha256_intersection':0,'train_dev_confirmed_same_group_intersection':0,'dev_group_count':8,'duplicate_entries':0,'all_paths_exist':True,'original_train_coverage':800,'official_val_test_included':False,'all_checks_passed':True}
 limits=['No complete acquisition sequence metadata: known manual scene isolation does not prove universal scene independence.','Only 60 priority relations adjudicated. Remaining full-review template rows are not different decisions.','Eight dev images give a small, nonrepresentative development estimate, not an independent final test.','Unreviewed retrieval associations are disclosed, not automatically merged or classified different.','Future independent training must use fresh random initialization, never EXP004A smoke weights.','Original val/test are excluded. The historical train/0209 versus val/0880 duplicate cannot cross this train/dev split because val is not included.']
 data={'version':'u3d_train_dev_v2_candidate','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_dataset':'datasets/U3D/lowlight/train','creation_reason':'User-reviewed priority same/different graph; one kept representative per confirmed connected scene group','status':'reviewed_development_candidate_known_relations_isolated','development_use_scope':'Eligible under the requested known-relation checks; review unresolved retrieval risks before making independence claims. No training launched.','train':sorted(train),'dev':sorted(dev),'quarantine_scene_related':sorted(quarantine),'deferred_candidates':[x for x in candidates if x['decision']=='defer'],'excluded_candidates':[x for x in candidates if x['decision']=='exclude'],'manual_scene_groups':groups,'manual_review_source':source_hashes,'manual_review_counts':dict(collections.Counter(x['decision'] for x in priority)),'known_limitations':limits,'checks':checks,'image_sha256':hashes,'unreviewed_dev_train_retrieval_relations':pending_cross,'prior_assistant_visual_cross_boundary_records':old_visual,'original_train_duplicate_groups':[v for v in duplicates.values() if len(v)>1],'generator_sha256':sha(Path(__file__))}
 assert all(sha(ROOT/p)==h for p,h in source_hashes.items()),'Review changed during build'
 text='# U3D train/dev v2 candidate split\n\n'
 text+='This is a development split with human review and isolation of known scene relations. Missing acquisition-sequence metadata prevents claiming that every train/dev image is scene-independent. No training or model selection was performed.\n\n'
 text+=f'## Review and counts\n\n60 priority pairs: 23 same, 37 different, 0 uncertain/pending. CSV columns are consistent; no format repairs. Candidate decisions: 8 keep, 9 exclude, 23 defer. Original train: 800. Final train: {len(train)}; dev: {len(dev)}; quarantine: {len(quarantine)}. Total removed from training: {len(isolated)} (dev plus quarantine).\n\n'
 text+='## Complete known manual same groups\n\n| ID | Dev representative | Group size | All original-train members |\n|---|---|---:|---|\n'
 for g in groups:text+=f"| {g['candidate_id']} | {g['representative']} | {g['size']} | {', '.join(Path(p).name for p in g['members'])} |\n"
 text+='\nGroup size includes the representative. Group membership is complete for the supplied same graph, not a claim of complete physical-scene retrieval. Evidence edges and original paths are preserved in JSON. All 9 exclude representatives are linked to kept scenes and quarantined. Defer alone does not exclude a file: only connections to kept groups do.\n\n## Checks\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n\n'
 text+='SHA256 was computed from all 800 actual original train files, not assumed from cached metadata. Existing train-internal duplicate groups remain within train, not across dev. Original val/test are never split inputs. Old v1 manifests and all original images remain unchanged.\n\n'
 text+=f'## Outstanding review risk\n\n{len(pending_cross)} directed retrieval relations connect kept representatives to proposed train images without a priority judgment; {len(old_visual)} previous assistant visual records cross the known isolated-scene/train boundary. These are disclosed in the JSON, not new human same/different decisions. The 671-row full association CSV remains a historical pending template; the 60 priority decisions are authoritative here.\n\n'
 for l in limits:text+='- '+l+'\n'
 text+='\nThe requested automatic known-relation checks pass. This candidate can support development under the stated limitations; it is not an independent benchmark split. Resolve any subsequently confirmed cross-boundary relation by revising the split before further selection.\n\n## Provenance and reproduction\n\nRun `python -B scripts/build_u3d_train_dev_v2.py` from this repository using the project environment. Refuses overwrite and aborts on a failed assertion before writing split files. No dependencies installed. Input CSVs are not rewritten. The JSON contains review SHA256, per-image SHA256, group evidence and pending relations. Text lists contain only project-relative original paths. No commit or push.\n'
 payload=[json.dumps(data,ensure_ascii=False,indent=2)+'\n','\n'.join(sorted(train))+'\n','\n'.join(sorted(dev))+'\n','\n'.join(sorted(quarantine))+'\n',text]
 for p,s in zip(outputs,payload):
  with p.open('x',encoding='utf-8') as f:f.write(s)
 print(json.dumps({'train':len(train),'dev':len(dev),'quarantine':len(quarantine),'groups':[(g['candidate_id'],g['size']) for g in groups],'pending_cross':len(pending_cross),'old_visual_cross':old_visual,'checks':checks},indent=2))
if __name__=='__main__':main()
