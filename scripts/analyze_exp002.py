"""EXP002 audit and fixed-weight train/val diagnosis. No training or input writes."""
import argparse, csv, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f: json.dump(data, f, indent=2, ensure_ascii=False)
def save(p, im):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('xb') as f: im.save(f, format='PNG')
def luminance(a):
    return np.asarray(a,dtype=np.float64) @ np.array([.2126,.7152,.0722]) / 255

def stats(a):
    y=luminance(a)
    dx=np.abs(np.diff(y,axis=1)).mean(); dy=np.abs(np.diff(y,axis=0)).mean()
    residual=y[1:-1,1:-1]-(y[:-2,1:-1]+y[2:,1:-1]+y[1:-1,:-2]+y[1:-1,2:])/4
    return dict(mean_y=float(y.mean()),std_y=float(y.std()),p05=float(np.quantile(y,.05)),p50=float(np.median(y)),p95=float(np.quantile(y,.95)),dark_y_lt_01=float((y<.1).mean()),bright_y_gt_09=float((y>.9).mean()),any_channel_255=float((a==255).any(2).mean()),all_channels_255=float((a==255).all(2).mean()),channel_255=float((a==255).mean()),gradient_l1=float((dx+dy)/2),highpass_std=float(residual.std()))

def histogram_svg(p, a,b):
    hist=[np.histogram(luminance(x),bins=64,range=(0,1))[0]/(x.shape[0]*x.shape[1]) for x in (a,b)]
    mx=max(float(h.max()) for h in hist)
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="400" viewBox="0 0 900 400"><rect width="900" height="400" fill="white"/><text x="60" y="25">Encoded RGB luma histogram (64 bins; fraction per bin)</text><path d="M60 50V340H850" stroke="black" fill="none"/>']
    for h,color,label,offset in zip(hist,['#2471a3','#d35400'],['Input','Enhanced'],[0,110]):
        points=' '.join(f'{60+(i+.5)*790/64:.2f},{340-v/mx*280:.2f}' for i,v in enumerate(h))
        s.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/><text x="{600+offset}" y="45" fill="{color}">{label}</text>')
    s.extend([f'<text x="20" y="60">{mx:.3f}</text>','<text x="60" y="365">0</text><text x="840" y="365">1</text></svg>'])
    with p.open('x',encoding='utf-8') as f:f.write(''.join(s))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--experiment',default='exp002_baseline_diagnosis'); args=ap.parse_args()
    root=args.root.resolve(); exp=(root/'experiments'/args.experiment).resolve()
    if not exp.is_relative_to((root/'experiments').resolve()) or exp.exists():raise ValueError('Use a new experiment folder inside experiments; overwrite disabled')
    # Inspect only explicit train/val directories. Test is used exclusively for EXP001 integrity audit below.
    exp.mkdir(parents=True);out=exp/'outputs';out.mkdir(); met=exp/'metrics';met.mkdir()
    spec=importlib.util.spec_from_file_location('exp001_entry',root/'scripts/infer_di_retinex.py');entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
    repo=root/'_model_audit/DI-retinex-low-light-enhancement'; source=repo/'mymodel.py';weight=root/'checkpoints/di_retinex/latest (21.54 lolv1).pth'
    assert entry.git(repo,'rev-parse','HEAD')==entry.COMMIT
    assert entry.git(repo,'hash-object',str(source))==entry.git(repo,'rev-parse','HEAD:mymodel.py')
    assert digest(weight)==entry.WEIGHT_SHA256
    import torch
    assert torch.cuda.is_available(), 'CUDA required'
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('official',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    net=mod.enhance_net_nopool(ch=6,number_f=64,linear=False).cuda().eval()
    net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
    def forward(a):
        x=torch.from_numpy(np.asarray(a,dtype=np.float32).copy()/255).permute(2,0,1).unsqueeze(0).contiguous().cuda()
        with torch.inference_mode():
            y,c,b,k=net(x)
            assert all(torch.isfinite(t).all().item() for t in (y,c,b,k))
            z=k*(x+.5*(b-1))+.5*(b+1)
            assert torch.allclose(y,z.clamp(0,1))
            info={'preclamp_gt1_channels':float((z>1).float().mean()),'preclamp_lt0_channels':float((z<0).float().mean()),'gain_mean':float(k.mean()),'gain_p95':float(torch.quantile(k.flatten(),.95))}
            fy=y[0].permute(1,2,0).cpu().numpy()
        return np.rint(fy*255).astype('uint8'),info,fy
    old=json.loads((root/'experiments/exp001_di_retinex/metrics/inference.json').read_text(encoding='utf-8'))
    original=Path(old['input']);before=digest(original)
    with Image.open(original) as im: resized=im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS)
    olddir=Path(old['output_dir'])
    with Image.open(olddir/'input_1280x720.png') as im: a=np.array(im.convert('RGB'))
    with Image.open(olddir/'enhanced_1280x720.png') as im: b=np.array(im.convert('RGB'))
    assert np.array_equal(a,np.array(resized));assert before==old['input_sha256_before']
    for name in old['saved_images']: assert digest(olddir/name)==old['saved_images'][name]['sha256']
    rerun,info,fy=forward(resized)
    delta=np.abs(rerun.astype('int16')-b.astype('int16'))
    official_round=np.clip(fy*255+.5,0,255).astype('uint8')
    audit={'input_resize_exact':True,'original_sha256':before,'source_commit':entry.COMMIT,'source_sha256':digest(source),'weight_sha256':digest(weight),'replay_max_uint8_difference':int(delta.max()),'replay_different_channel_fraction':float((delta>0).mean()),'np_rint_vs_torchvision_round_max':int(np.abs(rerun.astype('int16')-official_round.astype('int16')).max()),'reconstructed_forward_matches':True,**info}
    assert delta.max()<=1,'Unexpected replay discrepancy'
    # Fixed coordinates in resized image; these crops are implementation audit only.
    rois={'dark_roof':(80,40,400,240),'lit_facade':(430,320,750,520),'road_detail':(850,440,1170,640)}
    allstats={'full':{'input':stats(a),'enhanced':stats(b)}}
    canvas=Image.new('RGB',(640,230*3),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,box) in enumerate(rois.items()):
        x1,y1,x2,y2=box;aa=a[y1:y2,x1:x2];bb=b[y1:y2,x1:x2]
        allstats[name]={'box_xyxy_exclusive':box,'input':stats(aa),'enhanced':stats(bb)}
        for kind,arr,j in [('input',aa,0),('enhanced',bb,1)]:
            save(out/f'{name}_{kind}.png',Image.fromarray(arr));canvas.paste(Image.fromarray(arr),(320*j,230*i+30))
        draw.text((5,230*i+5),name+' / input',fill='black');draw.text((325,230*i+5),'enhanced',fill='black')
    save(out/'exp001_aligned_crops.png',canvas);histogram_svg(met/'exp001_histogram.svg',a,b)
    hist=[]
    for kind,arr in [('input',a),('enhanced',b)]:
        counts,edges=np.histogram(luminance(arr),bins=64,range=(0,1))
        hist.extend(dict(image=kind,left=float(edges[i]),right=float(edges[i+1]),count=int(c),fraction=float(c/len(a.reshape(-1,3)))) for i,c in enumerate(counts))
    with (met/'histogram.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=hist[0].keys());w.writeheader();w.writerows(hist)
    write_json(met/'exp001_audit.json',audit);write_json(met/'exp001_statistics.json',allstats)
    print('EXP001 replay and crops complete',audit,flush=True)
    selected=[];inventory=[]
    for split in ('train','val'):
        rows=[]
        files=sorted(p for p in (root/'datasets/U3D/lowlight'/split).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff'))
        for p in files:
            with Image.open(p) as im:
                size=im.size;im.draft('RGB',(320,180));thumb=im.convert('RGB').resize((320,180),Image.Resampling.LANCZOS);s=stats(np.array(thumb))
            rows.append(dict(split=split,source=p.relative_to(root).as_posix(),width=size[0],height=size[1],**s))
        inventory.extend(rows)
        keys=['mean_y','std_y','dark_y_lt_01','bright_y_gt_09','gradient_l1']
        f=np.array([[r[k] for k in keys] for r in rows]);lo=f.min(0);scale=np.maximum(f.max(0)-lo,1e-9);f=(f-lo)/scale
        picks=[]
        for ix in (int(np.argmin(f[:,0])),int(np.argmax(f[:,0])),int(np.argmax(f[:,3])),int(np.argmax(f[:,1]))):
            if ix not in picks:picks.append(ix)
        while len(picks)<10:
            dist=((f[:,None,:]-f[picks][None,:,:])**2).sum(2).min(1);dist[picks]=-1;picks.append(int(dist.argmax()))
        for ix in picks:
            r=rows[ix].copy();r['selection']='feature-extremes then farthest-point; 320x180 thumbnails'
            r['illumination_type']=('very_dark' if r['mean_y']<.12 else 'mixed_dark' if r['mean_y']<.3 else 'relatively_bright')+('_high_contrast' if r['std_y']>.18 else '')
            r['sha256']=digest(root/r['source']);selected.append(r)
        print(split,'scanned',len(rows),'selected',len(picks),flush=True)
    assert len(selected)==20 and all('/test/' not in r['source'] for r in selected)
    assert len({r['sha256'] for r in selected})==20,'Duplicate selected images'
    contact=Image.new('RGB',(1280,4*220),'white');draw=ImageDraw.Draw(contact);diags=[]
    for i,r in enumerate(selected):
        p=root/r['source'];key=r['split']+'_'+p.stem
        with Image.open(p) as im: thumb=im.convert('RGB').resize((320,180),Image.Resampling.LANCZOS);inp=im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS)
        contact.paste(thumb,(i%4*320,i//4*176)) if False else None
        # 4 columns x 5 rows, separate from the paired comparison below.
        enhanced,details,_=forward(inp)
        save(out/'diagnostic'/f'{key}_enhanced.png',Image.fromarray(enhanced))
        aa=np.array(inp);s0=stats(aa);s1=stats(enhanced)
        paired=Image.new('RGB',(640,180));paired.paste(thumb,(0,0));paired.paste(Image.fromarray(enhanced).resize((320,180),Image.Resampling.LANCZOS),(320,0));save(out/'diagnostic'/f'{key}_pair.png',paired)
        diags.append(dict(source=r['source'],split=r['split'],input=s0,enhanced=s1,**details))
        assert digest(p)==r['sha256'];print('diagnostic',i+1,key,flush=True)
    contact=Image.new('RGB',(1280,5*210),'white');draw=ImageDraw.Draw(contact)
    for i,r in enumerate(selected):
        p=root/r['source']
        with Image.open(p) as im:thumb=im.convert('RGB').resize((320,180),Image.Resampling.LANCZOS)
        xy=(i%4*320,i//4*210);contact.paste(thumb,(xy[0],xy[1]+30));draw.text(xy,r['split']+'/'+p.name+' '+r['illumination_type'],fill='black')
    save(out/'diagnostic_contact.png',contact)
    write_json(exp/'diagnostic_manifest.json',{'split_counts':{'train':10,'val':10},'selection_features':keys,'test_excluded':True,'scene_labels':'illumination statistics only; semantic labels in review document','images':selected})
    write_json(met/'train_val_inventory.json',inventory);write_json(met/'diagnostic_metrics.json',diags)
    assert digest(original)==before
    write_json(met/'provenance.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),source_commit=entry.COMMIT,weight_sha256=entry.WEIGHT_SHA256,analysis_script_sha256=digest(Path(__file__)),command_argv=[sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:]],training=False,selection_before_inference=True))
    print('COMPLETE',exp,flush=True)
if __name__=='__main__': main()
