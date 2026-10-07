"""Build v2.1 from two human review rounds; refuses overwrite, never writes original images."""
import csv,hashlib,json,collections,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:r=list(csv.reader(f))
 assert r and len(set(r[0]))==len(r[0]) and all(len(x)==len(r[0]) for x in r),'CSV format error'
 return [dict(zip(r[0],x)) for x in r[1:]]
def main():
 s=ROOT/'configs/splits';oldpath=s/'u3d_train_dev_v2.json';review=ROOT/'experiments/exp004b2_dev_train_review/review_dev_train.csv';priority=ROOT/'experiments/exp004b_dev_review/priority_review.csv'
 targets=[s/'u3d_train_dev_v2_1.json',s/'u3d_train_v2_1.txt',s/'u3d_dev_v2_1.txt',s/'u3d_quarantine_v2_1.txt',ROOT/'docs/U3D_TRAIN_DEV_V2_1.md']
 assert not any(p.exists() for p in targets),'Refuse overwrite'
 protected=[oldpath,review,priority,ROOT/'experiments/exp004b_dev_review/candidate_decisions.csv',ROOT/'docs/U3D_TRAIN_DEV_V2.md',*s.glob('u3d_*_v2.txt')]
 before={p.relative_to(ROOT).as_posix():sha(p) for p in protected}
 old=json.loads(oldpath.read_text(encoding='utf-8'));rows=read(review);first=read(priority)
 assert len(rows)==42 and collections.Counter(x['decision'] for x in rows)=={'different':36,'same':5,'uncertain':1}
 assert len({x['pair_id'] for x in rows})==42
 def pair(a,b):return tuple(sorted((a,b)))
 actual={pair(x['dev_image'],x['train_image']) for x in rows}
 expected={pair(x['candidate'],x['related_train']) for x in old['unreviewed_dev_train_retrieval_relations']}
 assert len(actual)==42 and actual==expected,'Review pairs do not match v2 pending relationships'
 representatives={x['candidate_id']:x['representative'] for x in old['manual_scene_groups']}
 train0=set(old['train']);dev=set(old['dev']);q0=set(old['quarantine_scene_related'])
 assert (len(train0),len(dev),len(q0))==(776,8,16)
 assert len(first)==60 and set(x['decision'] for x in first)<={'same','different','uncertain'}
 assert sha(priority)==old['manual_review_source'][priority.relative_to(ROOT).as_posix()],'Round one changed'
 for x in rows:assert x['dev_image']==representatives[x['dev_candidate_id']] and x['dev_image'] in dev and x['train_image'] in train0
 universe={p.relative_to(ROOT).as_posix() for p in (ROOT/'datasets/U3D/lowlight/train').iterdir() if p.is_file()}
 assert len(universe)==800 and universe==train0|dev|q0
 parent={p:p for p in universe}
 def find(p):
  if parent[p]!=p:parent[p]=find(parent[p])
  return parent[p]
 edges=[dict(a=x['path_a'],b=x['path_b'],decision=x['decision'],note=x['evidence_or_note'],round=1) for x in first]+[dict(a=x['dev_image'],b=x['train_image'],decision=x['decision'],note=x['note'],round=2,pair_id=x['pair_id']) for x in rows]
 for x in edges:
  assert x['a'] in universe and x['b'] in universe
  if x['decision']=='same':parent[find(x['a'])]=find(x['b'])
 assert not [x for x in edges if x['decision']=='different' and find(x['a'])==find(x['b'])],'same/different transitive conflict'
 assert len({find(p) for p in dev})==8,'Dev representatives merged'
 added={x['train_image'] for x in rows if x['decision'] in {'same','uncertain'}}
 train=train0-added;q=q0|added
 assert train and dev and not(train&dev or train&q or dev&q)
 assert train|dev|q==universe
 assert not({find(p) for p in train}&{find(p) for p in dev}),'Confirmed same group crosses train/dev'
 assert all(x['train_image'] in q for x in rows if x['decision']=='uncertain')
 assert all(x['train_image'] in train for x in rows if x['decision']=='different'),'Different conflicts with quarantine requirement'
 hashes={p:sha(ROOT/p) for p in sorted(universe)}
 assert hashes==old['image_sha256'],'Original train files changed since v2'
 assert not({hashes[p] for p in train}&{hashes[p] for p in dev}),'Byte duplicates across train/dev'
 assert not({hashes[p] for p in train}&{hashes[p] for p in q}),'Byte duplicates across train/quarantine'
 assert len({hashes[p] for p in dev})==8
 groups=[]
 for prior in old['manual_scene_groups']:
  members=sorted(p for p in universe if find(p)==find(prior['representative']))
  groups.append({**prior,'members':members,'size':len(members),'same_evidence':[x for x in edges if x['decision']=='same' and x['a'] in members]})
 checks={'all_passed':True,'csv_rows':42,'csv_columns_valid':True,'duplicate_review_pairs':0,'pending_user':0,'same_different_conflicts':0,'train_dev_path_intersection':0,'train_quarantine_path_intersection':0,'dev_quarantine_path_intersection':0,'coverage_original_train':800,'duplicate_entries_each_set':0,'all_files_exist':True,'train_dev_sha256_intersection':0,'train_quarantine_sha256_intersection':0,'train_dev_confirmed_same_group_intersection':0,'all_uncertain_targets_quarantined':True,'dev_representatives_unchanged':True,'original_images_match_v2_sha256':True}
 limitations=['Two rounds of targeted manual review do not prove absolute scene independence without acquisition sequence metadata.','The 42 selected dev-to-train relations are resolved for split handling, not an exhaustive new similarity search.','One uncertain pair remains semantically unresolved; its target is conservatively quarantined, not declared same.','Eight dev representatives form a small development set, not a final independent benchmark.','Original val/test remain excluded; fresh random initialization is required for independent experiments, without reusing smoke checkpoints.']
 data={k:v for k,v in old.items() if k not in ['unreviewed_dev_train_retrieval_relations','manual_review_counts','checks','generator_sha256','development_use_scope']}
 data.update(version='u3d_train_dev_v2_1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_version={'path':oldpath.relative_to(ROOT).as_posix(),'sha256':before[oldpath.relative_to(ROOT).as_posix()]},creation_reason='Round-two human same/uncertain train targets quarantined; no dev replacement',status='two_round_reviewed_development_split',development_use_scope='Eligible for development train/dev integration under documented limitations; no training performed',train=sorted(train),dev=old['dev'],quarantine_scene_related=sorted(q),quarantine_reasons=[{'image':x['train_image'],'decision':x['decision'],'dev_image':x['dev_image'],'candidate_id':x['dev_candidate_id'],'pair_id':x['pair_id'],'note':x['note']} for x in rows if x['decision'] in {'same','uncertain'}],manual_scene_groups=groups,manual_review_counts={'round1':dict(collections.Counter(x['decision'] for x in first)),'round2':dict(collections.Counter(x['decision'] for x in rows))},round2_review_records=rows,manual_relation_evidence=edges,unresolved_uncertain_relations=[x for x in rows if x['decision']=='uncertain'],remaining_unreviewed_from_v2_queue=[],known_limitations=limitations,checks=checks,protected_input_sha256=before,generator_sha256=sha(Path(__file__)))
 data['manual_review_source']={**old['manual_review_source'],review.relative_to(ROOT).as_posix():before[review.relative_to(ROOT).as_posix()]}
 text='# U3D train/dev v2.1\n\nTwo rounds of targeted human review have produced this development split. U3D lacks complete acquisition-sequence metadata; this does not prove that all potential scene relationships have been excluded.\n\n## Change from v2\n\n| Version | Train | Dev | Quarantine |\n|---|---:|---:|---:|\n| v2 |776|8|16|\n| v2.1 |'+str(len(train))+'|8|'+str(len(q))+'|\n\n42 review rows: 36 different, 5 same, 1 uncertain, no pending or duplicate pair. CSV column counts are consistent. Decisions and notes are preserved verbatim in JSON; no missing notes were inferred. Old v2 is unchanged. All six new quarantine files were original v2 train members.\n\n## Newly quarantined\n\n| Pair | Dev | Image | Human decision | Handling |\n|---|---|---|---|---|\n'
 for x in rows:
  if x['decision']!='different':text+=f"| {x['pair_id']} | {x['dev_candidate_id']} / {Path(x['dev_image']).name} | {x['train_image']} | {x['decision']} | quarantine |\n"
 text+='\n0207.jpg is quarantined conservatively; uncertain is NOT a same edge and does not enlarge the confirmed D09 scene group. All 36 different targets remain in train.\n\n## Confirmed same connected components\n\n| ID | Representative | Size including dev | Complete known members |\n|---|---|---:|---|\n'
 for g in groups:text+=f"| {g['candidate_id']} | {Path(g['representative']).name} | {g['size']} | {', '.join(Path(p).name for p in g['members'])} |\n"
 text+='\nEvery confirmed same edge from both rounds is retained in the JSON with its source round and original note. Transitive closure has no different edge inside a same component. All eight representatives remain in separate confirmed components.\n\n## Leakage checks\n\n```json\n'+json.dumps(checks,indent=2)+'\n```\n\nSHA256 was recomputed from all 800 actual images and matched v2. The three disjoint lists cover exactly the original official train. Official val/test are not used. Historical train/0209 versus val/0880 duplication cannot cross this train/dev boundary because official val is excluded.\n\n## Conditions and limitations\n\nThe requested checks pass: v2.1 is ready for development train/dev integration, not a claim of full benchmark independence. No training or model loading was performed.\n\n'+''.join('- '+v+'\n' for v in limitations)
 text+='\n## Reproduction and provenance\n\n`python -B scripts/build_u3d_train_dev_v2_1.py` uses standard-library code and refuses overwriting outputs. JSON includes input CSV/parent split hashes, all image hashes, exact review notes, same edges and quarantine reasons. Old v2 and human CSVs are never rewritten. The four new manifests use project-relative image paths. No dependencies, environment changes, commit or push.\n'
 assert all(sha(ROOT/p)==h for p,h in before.items()),'Inputs changed during generation'
 contents=[json.dumps(data,ensure_ascii=False,indent=2)+'\n','\n'.join(sorted(train))+'\n','\n'.join(old['dev'])+'\n','\n'.join(sorted(q))+'\n',text]
 for p,t in zip(targets,contents):
  with p.open('x',encoding='utf-8') as f:f.write(t)
 print(json.dumps({'train':len(train),'dev':len(dev),'quarantine':len(q),'added':sorted(added),'groups':[(g['candidate_id'],g['size']) for g in groups],'checks':checks},indent=2))
if __name__=='__main__':main()
