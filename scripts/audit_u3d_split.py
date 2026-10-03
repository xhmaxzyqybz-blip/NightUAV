import argparse,hashlib,json,csv
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();exp=root/'experiments/exp003_di_retinex_smoke';out=exp/'outputs/split_audit';met=exp/'audit';out.mkdir(parents=True,exist_ok=False);met.mkdir()
def dump(name,x):
 with (met/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];thumbs=[];bits=[];features=[]
t=np.arange(32);D=np.cos(np.pi*(2*t[None,:]+1)*t[:,None]/64);D[0]*=1/np.sqrt(2);D*=np.sqrt(2/32)
structure={}
for illumination in ['lowlight','light']:
 structure[illumination]={d.name:len([p for p in d.iterdir() if p.is_file()]) for d in (root/'datasets/U3D'/illumination).iterdir() if d.is_dir()}
for split in ['train','val']:
 for path in sorted((root/'datasets/U3D/lowlight'/split).glob('*')):
  if path.suffix.lower() not in ['.jpg','.jpeg','.png']:continue
  h=sha(path)
  with Image.open(path) as im:
   size=im.size;im.draft('RGB',(320,180));im=im.convert('RGB');thumb=im.resize((320,180),Image.Resampling.LANCZOS);gray=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
  coef=(D@gray@D.T)[:8,:8].flatten()[1:];bit=coef>np.median(coef)
  f=gray.flatten();f=f-f.mean();f=f/max(np.linalg.norm(f),1e-12)
  rows.append({'source':path.relative_to(root).as_posix(),'split':split,'sha256':h,'width':size[0],'height':size[1],'phash63':''.join('1' if x else '0' for x in bit)})
  thumbs.append(thumb);bits.append(bit);features.append(f)
  if len(rows)%100==0:print('audited',len(rows),flush=True)
n=sum(r['split']=='train' for r in rows);B=np.array(bits);F=np.array(features);hd=(B[:n,None,:]!=B[None,n:,:]).sum(2);corr=F[:n]@F[n:].T
exact={}
for r in rows:exact.setdefault(r['sha256'],[]).append(r['source'])
groups=[v for v in exact.values() if len(v)>1]
pairs=[]
for i,j in np.argwhere((hd<=10)|(corr>=.97)):
 pairs.append({'train':rows[i]['source'],'val':rows[n+j]['source'],'phash_hamming':int(hd[i,j]),'gray_ncc':float(corr[i,j]),'exact_bytes':rows[i]['sha256']==rows[n+j]['sha256'],'review':'pending'})
pairs.sort(key=lambda x:(x['phash_hamming'],-x['gray_ncc']))
dump('inventory.json',rows);dump('exact_duplicates.json',groups);dump('candidates.json',pairs)
with (met/'candidates.csv').open('x',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['train','val','phash_hamming','gray_ncc','exact_bytes','review']);w.writeheader();w.writerows(pairs)
lookup={r['source']:i for i,r in enumerate(rows)}
manual=[('0008','0815'),('0595','0894')]
view=pairs[:40].copy()
for tr,va in manual:
 tr=f'datasets/U3D/lowlight/train/{tr}.jpg';va=f'datasets/U3D/lowlight/val/{va}.jpg';i=lookup[tr];j=lookup[va]-n
 if not any(x['train']==tr and x['val']==va for x in view):view.append(dict(train=tr,val=va,phash_hamming=int(hd[i,j]),gray_ncc=float(corr[i,j]),exact_bytes=rows[i]['sha256']==rows[n+j]['sha256'],review='previously flagged'))
# Most dissimilar candidates are only smoke eligibility candidates, not proven scene-independent.
allowed=[i for i in range(n) if int(hd[i].min())>10 and float(corr[i].max())<.90]
allowed.sort(key=lambda i:float(corr[i].max()))
smoke=[]
for i in allowed[:4]:
 j=int(corr[i].argmax());row={**rows[i],'nearest_val_ncc':rows[n+j]['source'],'max_val_ncc':float(corr[i,j]),'min_val_phash_distance':int(hd[i].min())};smoke.append(row)
 view.append(dict(train=rows[i]['source'],val=rows[n+j]['source'],phash_hamming=int(hd[i,j]),gray_ncc=float(corr[i,j]),review='smoke_nearest_visual_check'))
for page in range((len(view)+5)//6):
 canvas=Image.new('RGB',(1280,630),'white');draw=ImageDraw.Draw(canvas)
 for z,row in enumerate(view[page*6:page*6+6]):
  x=(z%2)*640;y=(z//2)*210
  for off,key in [(0,'train'),(320,'val')]:canvas.paste(thumbs[lookup[row[key]]],(x+off,y+30))
  draw.text((x,y),f"#{page*6+z} {Path(row['train']).stem}/{Path(row['val']).stem} d={row['phash_hamming']} ncc={row['gray_ncc']:.3f}",fill='black')
 with (out/f'candidates_{page:02d}.png').open('xb') as f:canvas.save(f,format='PNG')
for num,row in enumerate(view):
 canvas=Image.new('RGB',(640,210),'white');draw=ImageDraw.Draw(canvas)
 canvas.paste(thumbs[lookup[row['train']]],(0,30));canvas.paste(thumbs[lookup[row['val']]],(320,30));draw.text((2,2),f"#{num} {Path(row['train']).stem}/{Path(row['val']).stem} d={row['phash_hamming']} ncc={row['gray_ncc']:.3f}",fill='black')
 with (out/f'pair_{num:03d}.png').open('xb') as f:canvas.save(f,format='PNG')
dump('visualized_pairs.json',view);dump('smoke_candidates.json',smoke)
summary=dict(directory_counts=structure,audited_train=n,audited_val=len(rows)-n,comparisons=n*(len(rows)-n),exact_duplicate_groups=len(groups),exact_cross_split_groups=sum(any('/train/' in x for x in v) and any('/val/' in x for x in v) for v in groups),candidate_pairs=len(pairs),candidate_train_images=len(set(x['train'] for x in pairs)),candidate_val_images=len(set(x['val'] for x in pairs)),visualized_pairs=len(view),threshold='63-bit DCT hash Hamming<=10 OR 32x32 grayscale NCC>=0.97',smoke_eligible=len(allowed),script_sha256=sha(Path(__file__)))
dump('summary.json',summary);print(json.dumps(summary),flush=True)
