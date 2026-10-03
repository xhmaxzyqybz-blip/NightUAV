"""Validate file/group isolation separately from scene-review release readiness."""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter

def validate(root,verify_hashes=True):
 s=root/'configs/splits';meta=json.loads((s/'u3d_scene_v1_metadata.json').read_text(encoding='utf-8'));manifest=json.loads((s/'u3d_scene_groups_v1.json').read_text(encoding='utf-8'));rows={r['source']:r for r in manifest['records']};errors=[];lists={};stats={}
 if len(rows)!=len(manifest['records']):errors.append('duplicate manifest source')
 for name,expected in meta['files'].items():
  if hashlib.sha256((s/name).read_bytes()).hexdigest()!=expected:errors.append('manifest hash mismatch '+name)
 for split in ['train','dev','quarantine']:
  items=[x.strip() for x in (s/f'u3d_scene_{split}_v1.txt').read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')];lists[split]=items
  if len(items)!=len(set(items)):errors.append('duplicate path in '+split)
  normalized=[]
  for path in items:
   p=(root/path).resolve();normalized.append(str(p).casefold())
   if not p.is_relative_to((root/'datasets/U3D/lowlight/train').resolve()):errors.append('non-original-train '+path)
   if not p.is_file():errors.append('missing '+path);continue
   if path not in rows:errors.append('untraceable '+path);continue
   r=rows[path]
   if r['original_split']!='train' or r['usage']!=split:errors.append('source/usage mismatch '+path)
   if verify_hashes and hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:errors.append('data hash changed '+path)
  if len(set(normalized))!=len(normalized):errors.append('canonical path collision '+split)
  selected=[rows[x] for x in items if x in rows];stats[split]={'images':len(items),'scene_groups':len({x['scene_group_id'] for x in selected}),'conservative_groups':len({x['conservative_group_id'] for x in selected}),'scene_categories':dict(Counter(x['scene_category'] for x in selected)),'confirmation_status':dict(Counter(x['confirmation_status'] for x in selected))}
 for a,b in [('train','dev'),('train','quarantine'),('dev','quarantine')]:
  if set(lists[a])&set(lists[b]):errors.append('file overlap '+a+'/'+b)
  for key in ['scene_group_id','conservative_group_id','sha256']:
   if {rows[x][key] for x in lists[a]} & {rows[x][key] for x in lists[b]}:errors.append(key+' overlap '+a+'/'+b)
 actual={p.relative_to(root).as_posix() for p in (root/'datasets/U3D/lowlight/train').iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']}
 if set().union(*(set(v) for v in lists.values()))!=actual:errors.append('original train not covered exactly')
 if len(actual)!=800:errors.append('unexpected original train size')
 release=meta.get('training_allowed') is True and manifest.get('training_allowed') is True and meta['status']=='approved' and len(lists['train'])>0 and len(lists['dev'])>0
 return {'structural_checks_passed':not errors,'ready_for_training':not errors and release,'errors':errors,'readiness_blockers':meta.get('blocking_reasons',[])+([] if lists['train'] and lists['dev'] else ['train/dev empty; intentional quarantine-only draft']),'stats':stats,'data_hashes_checked':verify_hashes,'limitation':'Empty train/dev make non-overlap checks vacuously true; not proof of independent scenes. Metadata review release required.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--require-ready',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args();result=validate(a.root.resolve())
 if a.report:
  target=a.report.resolve()
  if not target.is_relative_to((a.root/'experiments').resolve()):raise ValueError('Report must be under experiments')
  with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2));return 0 if result['structural_checks_passed'] and (not a.require_ready or result['ready_for_training']) else 2
if __name__=='__main__':sys.exit(main())
