"""Read-only exact RGB-zero crop audit. No model imports or training changes."""
import csv, hashlib, io, json, random, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/exp005a_black_crop_audit'
DOC=ROOT/'docs/U3D_BLACK_CROP_AUDIT.md'
PATCH=256
SEEDS=[1149,2026,3407]
DRAWS=1000

def black_windows(rgb,k):
    occupied=np.any(rgb!=0,axis=2).astype(np.uint8)
    sat=cv2.integral(occupied,sdepth=cv2.CV_32S)
    sums=sat[k:,k:]-sat[:-k,k:]-sat[k:,:-k]+sat[:-k,:-k]
    return sums==0,int(occupied.size-np.count_nonzero(occupied))

def check_algorithm():
    rng=np.random.default_rng(11)
    cases=[np.zeros((9,11,3),np.uint8),np.ones((9,11,3),np.uint8)]
    for _ in range(10):
        a=rng.integers(0,2,(9,11,3),dtype=np.uint8);a[3:8,5:11]=0;cases.append(a)
    for a in cases:
        got,_=black_windows(a,4)
        expected=np.array([[not a[y:y+4,x:x+4].any() for x in range(8)] for y in range(6)])
        assert np.array_equal(got,expected)

def dump(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    check_algorithm()
    assert not OUT.exists() and not DOC.exists(),'Refuse to overwrite audit outputs'
    manifest=ROOT/'configs/splits/u3d_train_v2_1.txt'
    manifest_bytes=manifest.read_bytes();paths=manifest_bytes.decode('utf-8').splitlines()
    assert len(paths)==len(set(paths))==770
    other=set()
    for n in ['dev','quarantine']:other.update((ROOT/f'configs/splits/u3d_{n}_v2_1.txt').read_text(encoding='utf-8').splitlines())
    assert not set(paths)&other
    expected=json.loads((ROOT/'configs/splits/u3d_train_dev_v2_1.json').read_text(encoding='utf-8'))['image_sha256']
    OUT.mkdir();rngs={s:random.Random(s) for s in SEEDS};records=[];started=time.perf_counter()
    for index,source in enumerate(paths,1):
        path=ROOT/source
        assert path.resolve().is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve())
        data=path.read_bytes();digest=hashlib.sha256(data).hexdigest();assert digest==expected[source]
        with Image.open(io.BytesIO(data)) as im:rgb=np.array(im.convert('RGB'))
        h,w=rgb.shape[:2];assert h>=PATCH and w>=PATCH and h*w<2**31
        valid,zeros=black_windows(rgb,PATCH);count=int(np.count_nonzero(valid));total=valid.size;groups=[]
        if count:
            n,labels,stats,centroids=cv2.connectedComponentsWithStats(valid.astype(np.uint8),connectivity=8)
            for label in range(1,n):
                x,y,bw,bh,area=map(int,stats[label]);dy,dx=np.argwhere(labels[y:y+bh,x:x+bw]==label)[0];rx=x+int(dx);ry=y+int(dy)
                assert not rgb[ry:ry+PATCH,rx:rx+PATCH].any()
                groups.append({'top_left_component_id':label,'valid_top_left_count':area,'top_left_bbox_xywh':[x,y,bw,bh],'covered_pixel_bbox_xywh':[x,y,bw+PATCH-1,bh+PATCH-1],'representative_crop_xywh':[rx,ry,PATCH,PATCH]})
            assert sum(g['valid_top_left_count'] for g in groups)==count
            del labels,stats,centroids
        samples={}
        for seed in SEEDS:
            r=rngs[seed];hits=0
            for _ in range(DRAWS):
                x=r.randint(0,w-PATCH);y=r.randint(0,h-PATCH);hits+=int(valid[y,x])
            samples[str(seed)]={'draws':DRAWS,'black_crops':hits,'rate':hits/DRAWS}
        known=None
        if Path(source).name=='0069.jpg':known=bool(valid[1850,68]);assert known
        records.append({'source':source,'filename':Path(source).name,'sha256':digest,'width':w,'height':h,'black_pixel_count':zeros,'black_pixel_ratio':zeros/(w*h),'nonzero_pixel_ratio':1-zeros/(w*h),'has_black_crop':bool(count),'black_crop_top_left_count':count,'possible_crop_top_left_count':total,'exact_uniform_crop_probability':count/total,'top_left_component_count':len(groups),'components':groups,'random_samples':samples,'known_failed_crop_68_1850_is_black':known})
        if index%50==0 or index==len(paths):print(json.dumps({'images_done':index,'images_with_black_crop':sum(x['has_black_crop'] for x in records),'seconds':round(time.perf_counter()-started,2)}),flush=True)
    assert manifest.read_bytes()==manifest_bytes
    affected=[x for x in records if x['has_black_crop']]
    totals={str(s):{'draws':len(records)*DRAWS,'black_crops':sum(x['random_samples'][str(s)]['black_crops'] for x in records),'images_with_sampled_black_crop':sum(x['random_samples'][str(s)]['black_crops']>0 for x in records)} for s in SEEDS}
    for v in totals.values():v['rate']=v['black_crops']/v['draws']
    hit=sum(v['black_crops'] for v in totals.values());draws=len(records)*len(SEEDS)*DRAWS
    summary={'images':len(records),'images_with_black_crop':len(affected),'images_with_sampled_black_crop_any_seed':sum(any(z['black_crops'] for z in x['random_samples'].values()) for x in records),'random_black_crops':hit,'random_draws':draws,'random_global_rate':hit/draws,'exact_uniform_image_then_crop_rate':sum(x['exact_uniform_crop_probability'] for x in records)/len(records),'per_seed':totals,'0069':next(x for x in records if x['filename']=='0069.jpg'),'seconds':time.perf_counter()-started,'all_image_hashes_match_split':True,'algorithm_selfcheck_passed':True}
    dump(OUT/'summary.json',summary);dump(OUT/'per_image.json',records)
    dump(OUT/'config.json',{'patch':PATCH,'seeds':SEEDS,'draws_per_image_per_seed':DRAWS,'manifest':manifest.relative_to(ROOT).as_posix(),'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'decode':'Pillow convert RGB; uint8 exact equality to zero, matching original loader RGB decode','algorithm':'int32 integral of any-channel-nonzero mask; all legal integer top-left positions, stride 1','rng':'independent Python random.Random per seed, continuously advanced in manifest order; x then y inclusive randint','numpy':np.__version__,'opencv':cv2.__version__})
    with (OUT/'per_image.csv').open('x',encoding='utf-8',newline='') as f:
        names=['source','width','height','has_black_crop','black_pixel_ratio','nonzero_pixel_ratio','black_crop_top_left_count','possible_crop_top_left_count','exact_uniform_crop_probability','top_left_component_count']+[f'seed{s}_black_crops' for s in SEEDS]
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader()
        for x in records:
            row={k:x[k] for k in names if k in x};row.update({f'seed{s}_black_crops':x['random_samples'][str(s)]['black_crops'] for s in SEEDS});writer.writerow(row)
    report='# U3D train v2.1 strict-zero 256x256 crop audit\n\nNo training, model loading, loss changes or original-image writes. All 770 source hashes match the existing split manifest.\n\n## Method and interpretation\n\nPillow RGB decoding matches the training loader. A black pixel has all three decoded uint8 RGB channels exactly zero. An integral image of the any-channel-nonzero mask tests EVERY legal 256x256 integer crop with stride 1, including bottom/right borders. No sampling is used to decide whether a black patch exists. Small brute-force tests independently verify integral-window indexing.\n\nRandom simulation: three independent Python Random streams (1149,2026,3407), 1000 draws per image per seed, manifest order, uniform inclusive x then y. These are risk simulations, not a replay of a particular training epoch. The exact probability counts all black-valid top-left positions / all legal positions; the dataset probability averages images equally.\n\nComponents are 8-connected clusters of VALID CROP TOP-LEFT coordinates. They identify separate feasible location clusters, not proven separate physical scenes. Bounding rectangles may contain holes and must not be interpreted as entirely black rectangles. Overlapping black crops are counted individually for sampling probability. Nonzero-pixel fraction describes available nonblack pixels; it does not certify semantic usefulness, exposure quality, or a fully nonblack 256x256 crop.\n\n## Results\n\n'
    report+=f"- Images containing at least one strict-zero patch: {len(affected)}/770.\n- Images hit by random simulation (union of seeds): {summary['images_with_sampled_black_crop_any_seed']}.\n- Sampled black crops: {hit}/{draws} ({100*hit/draws:.8f}%).\n- Exact equal-image uniform-crop risk: {100*summary['exact_uniform_image_then_crop_rate']:.8f}%.\n- Runtime: {summary['seconds']:.2f} seconds.\n\n"
    report+='|Source|Black pixels %|Nonzero pixels %|Black-valid positions|Exact crop risk %|Location clusters|Seed1149/2026/3407 hits (each /1000)|\n|---|---:|---:|---:|---:|---:|---|\n'
    for x in affected:report+=f"|{x['filename']}|{100*x['black_pixel_ratio']:.4f}|{100*x['nonzero_pixel_ratio']:.4f}|{x['black_crop_top_left_count']}|{100*x['exact_uniform_crop_probability']:.6f}|{x['top_left_component_count']}|"+'/'.join(str(x['random_samples'][str(s)]['black_crops']) for s in SEEDS)+'|\n'
    report+='\n## Location details\n\nCoordinates use a zero-based top-left origin; boxes are x,y,width,height. Full per-image records are in experiments/exp005a_black_crop_audit/per_image.json.\n\n'
    for x in affected:
        report+=f"### {x['filename']}\n\n"
        for g in x['components']:report+=f"- Valid-top-left cluster {g['top_left_component_id']}: {g['top_left_bbox_xywh']}; representative exact black crop {g['representative_crop_xywh']}; {g['valid_top_left_count']} positions.\n"
    z=summary['0069'];zh=sum(v['black_crops'] for v in z['random_samples'].values())
    report+=f"\n## 0069.jpg\n\nPreviously failed crop (68,1850,256,256) independently confirmed strict-zero. Exact random-crop probability: {100*z['exact_uniform_crop_probability']:.6f}%; simulation: {zh}/3000 ({100*zh/3000:.6f}%). Nonzero pixels: {100*z['nonzero_pixel_ratio']:.4f}%.\n\n## Limits and decision\n\nThis audit covers only the 770 training images, not dev, quarantine or official val/test. Strict zero differs from near-black; tiny positive means and other numerical risks are not audited. Random samples can miss rare black patches; the exhaustive check is authoritative for existence. Counts characterize current decoded RGB and the current 256x256 crop protocol. No crop policy is changed. Await human review before skipping, resampling, changing the objective, or restarting training.\n"
    with DOC.open('x',encoding='utf-8') as f:f.write(report)
    with (OUT/'README.md').open('x',encoding='utf-8') as f:f.write('# Strict-zero crop audit\n\nReport: ../../docs/U3D_BLACK_CROP_AUDIT.md\n\nRun with nightuav Python: `python -B scripts/audit_u3d_black_crops.py`. The script refuses to overwrite existing outputs. Summary, per-image JSON/CSV and configuration are retained here. No model or training is run.\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()