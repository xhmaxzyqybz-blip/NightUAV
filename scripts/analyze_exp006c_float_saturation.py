"""EXP006C: read-only frozen-model diagnostics. No training or source mutation."""
import hashlib, importlib.util, json, sys, time, traceback, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'experiments/exp006c_float_saturation_attribution'
BASE = ROOT/'experiments/exp005b_di_retinex_u3d_baseline'
PREV = ROOT/'experiments/exp006a_failure_analysis'
REPO = ROOT/'_model_audit/DI-retinex-low-light-enhancement'
CHECKPOINT = ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth'
EXPECTED = '73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0'
COEF = np.array([.2126,.7152,.0722],dtype=np.float32)

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def dist(a):
    a=np.asarray(a,dtype=np.float64).ravel()
    if not a.size: return {'count':0,'mean':None}
    qs=[0,.01,.1,.5,.9,.95,.99,1]
    return {'count':a.size,'mean':float(a.mean()),'std':float(a.std()),'quantile_levels':qs,'quantiles':np.quantile(a,qs).tolist()}
def histogram(a):
    a=np.asarray(a).ravel()
    edges=np.unique(np.r_[min(float(a.min()),-1e-6),0,np.linspace(0,1,51),max(float(a.max()),1.000001)])
    count,edge=np.histogram(a,edges)
    return {'edges':edge.tolist(),'counts':count.tolist(),'scope':'full range, no discarded tails'}
def metric(a,denom=None):
    return {'count':int(np.count_nonzero(a)),'denominator':int(a.size if denom is None else denom),
            'ratio':float(np.count_nonzero(a)/(a.size if denom is None else denom)) if (a.size if denom is None else denom) else None}
def summarize(x,z,y,q,mask):
    x=x[mask];z=z[mask];y=y[mask];q=q[mask]
    gt=z>1;lt=z<0;eq=z==1;roundonly=(z>0)&(z<1)&(q==255)
    endpoint=q==255;floatend=y==1
    assert np.array_equal(endpoint,gt|eq|roundonly)
    n=len(x); d={}
    for name,event in [('pre_gt1',gt),('pre_lt0',lt),('pre_eq1',eq),('post_eq1',floatend),
                       ('png255',endpoint),('rounding_only',roundonly),('nonclipped_png255',(z>0)&(z<=1)&endpoint)]:
        d[name+'_channel']=metric(event)
        d[name+'_any_pixel']=metric(event.any(1))
        d[name+'_all_channels_pixel']=metric(event.all(1))
    # Exclusive causes for any endpoint: strict clipping takes priority, then exact-one, then rounding.
    for name,event in [('any_endpoint_strict_clip',gt.any(1)),
        ('any_endpoint_exact1_no_clip',~gt.any(1)&eq.any(1)),
        ('any_endpoint_rounding_only',~gt.any(1)&~eq.any(1)&endpoint.any(1)),
        ('white_with_any_strict_clip',endpoint.all(1)&gt.any(1)),
        ('white_no_clip_with_exact1',endpoint.all(1)&~gt.any(1)&eq.any(1)),
        ('white_all_rounding',endpoint.all(1)&~gt.any(1)&~eq.any(1)),
        ('white_rounding_assisted',endpoint.all(1)&~floatend.all(1)),
        ('white_all_float1',floatend.all(1))]:
        d[name]=metric(event)
    assert d['png255_any_pixel']['count']==sum(d[k]['count'] for k in ['any_endpoint_strict_clip','any_endpoint_exact1_no_clip','any_endpoint_rounding_only'])
    assert d['png255_all_channels_pixel']['count']==sum(d[k]['count'] for k in ['white_with_any_strict_clip','white_no_clip_with_exact1','white_all_rounding'])
    for label,after in [('pre_gt1',gt),('post_eq1',floatend),('png255',endpoint)]:
        for mode in ['channel','any_pixel','all_channels_pixel']:
            before=x==1
            if mode=='any_pixel': before=before.any(1);out=after.any(1)
            elif mode=='all_channels_pixel': before=before.all(1);out=after.all(1)
            else: out=after
            prefix=label+'_'+mode
            d[prefix+'_input_existing']=metric(before)
            d[prefix+'_new_over_region']=metric(~before & out)
            d[prefix+'_new_over_input_nonendpoint']=metric(~before & out,int((~before).sum()))
            d[prefix+'_retained']=metric(before & out)
            d[prefix+'_released']=metric(before & ~out)
    return {'pixels':n,'metrics':d}

def aggregate(rows):
    out={}
    for region in ['all','dark','middle','bright']:
        values=[r['regions'][region] for r in rows]
        keys=values[0]['metrics'].keys(); pooled={}; means={}
        for key in keys:
            ms=[v['metrics'][key] for v in values]; count=sum(m['count'] for m in ms); denom=sum(m['denominator'] for m in ms)
            pooled[key]={'count':count,'denominator':denom,'ratio':count/denom if denom else None}
            valid=[m['ratio'] for m in ms if m['ratio'] is not None]
            means[key]={'ratio':float(np.mean(valid)) if valid else None,'nonempty_images':len(valid)}
        out[region]={'pixels':sum(v['pixels'] for v in values),'pooled':pooled,'image_mean':means}
    return out

def profile_svg(path,series,title):
    colors=['#2166ac','#b2182b','#777777']
    w,h=780,270;left,top=60,35;pw,ph=690,180
    maximum=max(1.,max(float(np.max(s)) for s in series.values()))
    minimum=min(0.,min(float(np.min(s)) for s in series.values()))
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>',
           f'<text x="15" y="20" font-family="sans-serif" font-size="14">{title}</text>']
    for v in np.linspace(minimum,maximum,5):
        yy=top+ph*(maximum-v)/(maximum-minimum)
        parts.append(f'<path d="M{left} {yy} H{left+pw}" stroke="#ddd"/><text x="4" y="{yy+4}" font-size="12">{v:.3f}</text>')
    for j,(name,s) in enumerate(series.items()):
        pts=' '.join(f'{left+pw*i/(len(s)-1):.2f},{top+ph*(maximum-float(v))/(maximum-minimum):.2f}' for i,v in enumerate(s))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[j]}" stroke-width="1.5"/><text x="{left+j*210}" y="250" fill="{colors[j]}" font-size="13">{name}</text>')
    parts.append(f'<text x="{left}" y="231" font-size="12">0</text><text x="{left+pw-30}" y="231" font-size="12">127 px</text></svg>')
    path.write_text(''.join(parts),encoding='utf-8')

def roi_analysis(rois,cache):
    result=[];index=['<meta charset="utf-8"><title>EXP006C fixed ROI</title><h1>Fixed EXP006A ROIs</h1><p>RGB input | E15; profiles use encoded Yprime. No output-based selection.</p>']
    for r in rois:
        source=r['source'];a,z,y,q=cache[source];xx,yy,w,h=r['xywh'];sl=np.s_[yy:yy+h,xx:xx+w]
        x=a.astype(np.float32)/255.; xin=x[sl];zz=z[sl];out=y[sl]
        yi=(xin*COEF).sum(2);yo=(out*COEF).sum(2);yz=(zz*COEF).sum(2)
        stem=Path(source).stem+'_'+r['category']
        stats={'source':source,'category':r['category'],'xywh':r['xywh'],'original_xywh':r['original_3840x2160_xywh'],
          'z_distribution':dist(zz),'z_histogram':histogram(zz),'Y_distribution':dist(out),'Y_histogram':histogram(out),
          'input_luminance':dist(yi),'output_luminance':dist(yo),'luminance_gain':dist(yo-yi),
          'attribution':summarize(xin,zz,out,q[sl],np.ones((h,w),bool)),
          'profiles':{},'gradients':{}}
        for name,arr in [('input',yi),('post_clamp',yo),('pre_clamp',yz)]:
            gx=np.diff(arr,axis=1);gy=np.diff(arr,axis=0)
            stats['gradients'][name]={'abs_horizontal':dist(np.abs(gx)),'abs_vertical':dist(np.abs(gy)),
                'magnitude_interior':dist(np.hypot(gx[:-1,:],gy[:,:-1])),
                'exact_zero_neighbor_difference_ratio':float((np.count_nonzero(gx==0)+np.count_nonzero(gy==0))/(gx.size+gy.size))}
            stats['profiles'][name]={'center_row':arr[h//2,:].tolist(),'center_column':arr[:,w//2].tolist(),
                                     'column_mean':arr.mean(0).tolist(),'row_mean':arr.mean(1).tolist()}
        panel=Image.new('RGB',(w*2,h+24),'white');panel.paste(Image.fromarray(a[sl]),(0,24));panel.paste(Image.fromarray(q[sl]),(w,24))
        ImageDraw.Draw(panel).text((4,4),'input                 E15',fill='black');panel.save(EXP/'outputs'/f'{stem}.png')
        for kind in ['center_row','center_column','column_mean','row_mean']:
            series={name:np.array(stats['profiles'][name][kind]) for name in stats['profiles']}
            profile_svg(EXP/'outputs'/f'{stem}_{kind}.svg',series,f'{stem} {kind}; encoded Yprime')
        np.savez_compressed(EXP/'outputs'/f'{stem}_floats.npz',input=xin,z=zz,Y=out,png=q[sl])
        result.append(stats)
        index.append(f'<h2>{stem}: {r["xywh"]}</h2><img src="{stem}.png"/><br/>')
        index.extend(f'<img src="{stem}_{kind}.svg"/>' for kind in ['center_row','center_column','column_mean','row_mean'])
    (EXP/'outputs/index.html').write_text('\n'.join(index),encoding='utf-8')
    return result

def main():
    assert not EXP.exists(),'Refuse to overwrite existing analysis'
    cfg=read(BASE/'config/run.json')
    protected={**cfg['protected_hashes'],**cfg['split_sha256']}
    protected['scripts/train_di_retinex_exp005b.py']=cfg['script_sha256']
    protected['scripts/nightuav_crop_adapter.py']=cfg['crop_protocol']['adapter_sha256']
    protected[CHECKPOINT.relative_to(ROOT).as_posix()]=EXPECTED
    for path,h in protected.items(): assert sha(ROOT/path)==h,path
    splits={k:(ROOT/f'configs/splits/u3d_{k}_v2_1.txt').read_text().splitlines() for k in ['train','dev','quarantine']}
    assert [len(splits[k]) for k in splits]==[770,8,22]
    assert all(len(v)==len(set(v)) for v in splits.values())
    assert not(set(splits['train'])&set(splits['dev']) or set(splits['dev'])&set(splits['quarantine']) or set(splits['train'])&set(splits['quarantine']))
    dev=splits['dev'];assert all((ROOT/p).is_file() and '/lowlight/train/' in p for p in dev)
    rois=read(PREV/'config/crop_coordinates.json')
    assert len(rois)==12 and {Path(r['source']).stem for r in rois}=={'0049','0179','0758','0792'}
    sources={r['source']:r for r in read(PREV/'config/sources.json')}
    fixed={r['source']:r for r in read(BASE/'config/fixed_dev_inputs.json')}
    assert set(dev)==set(sources)==set(fixed)
    for p in dev:
        for key,hkey in [('source','source_sha256'),('saved_input_path','saved_input_sha256'),('e15_output_path','e15_output_sha256')]:
            assert sha(ROOT/sources[p][key])==sources[p][hkey]
            protected[sources[p][key]]=sources[p][hkey]
    for p in [PREV/'config/crop_coordinates.json',BASE/'config/fixed_dev_inputs.json']:
        protected[p.relative_to(ROOT).as_posix()]=sha(p)
    assert subprocess.run(['git','-c','safe.directory=E:/NightUAV','-C',str(ROOT),'check-ignore','-q',str(EXP/'outputs/probe.png')]).returncode==0
    assert torch.cuda.is_available(),'CUDA unavailable; no CPU fallback'
    assert str(torch.__version__)==cfg['torch']
    torch.manual_seed(1149);torch.cuda.manual_seed_all(1149)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    spec=importlib.util.spec_from_file_location('exp006c_official_model',REPO/'mymodel.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    assert state['epoch']==15
    model=mod.enhance_net_nopool().cuda()
    load_result=model.load_state_dict(state['model'],strict=True)
    assert not load_result.missing_keys and not load_result.unexpected_keys
    model.eval();before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    del state
    EXP.mkdir(parents=True);(EXP/'outputs').mkdir()
    dump(EXP/'config/run.json',{'checkpoint_sha256':EXPECTED,'protected_hashes':protected,'dev':dev,'size':[1280,720],
         'normalization':'RGB uint8 /255. (NumPy float64) -> torch float32; LANCZOS as EXP005B',
         'z_expression':'k*(x+0.5*(b-1))+0.5*(b+1), original operation order',
         'quantization':'np.rint(Y_float32*255).astype(uint8), same as baseline',
         'regions':'input encoded RGB Yprime .2126R+.7152G+.0722B; dark<.2 middle[.2,.7] bright>.7',
         'roi_source':(PREV/'config/crop_coordinates.json').relative_to(ROOT).as_posix(),
         'roi_profile_selection':'center row y+64, center column x+64 and all-row/column means; fixed before outputs',
         'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'weights_only':True,'strict':True,
         'script_sha256':sha(Path(__file__)),'no_training':True})
    dump(EXP/'config/roi_coordinates.json',rois)
    cache={};checks=[]
    # Gate ALL images before causal counting/ROI analysis.
    with torch.no_grad():
        for p in dev:
            with Image.open(ROOT/p) as im:
                assert im.size==(3840,2160)
                a=np.array(im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS))
            assert hashlib.sha256(a.tobytes()).hexdigest()==fixed[p]['resized_rgb_sha256']
            with Image.open(ROOT/sources[p]['saved_input_path']) as im: assert np.array_equal(a,np.array(im))
            x=torch.from_numpy(a/255.).float().permute(2,0,1).unsqueeze(0).contiguous().cuda()
            torch.cuda.synchronize();t=time.perf_counter()
            y,c,b,k=model(x);z=k*(x+.5*(b-1))+.5*(b+1)
            torch.cuda.synchronize()
            assert torch.equal(y,z.clamp(0,1)),p+' clamp mismatch'
            assert all(torch.isfinite(v).all().item() for v in [y,c,b,k,z])
            za=z[0].permute(1,2,0).cpu().numpy();ya=y[0].permute(1,2,0).cpu().numpy()
            q=np.rint(ya*255).astype('uint8')
            with Image.open(ROOT/sources[p]['e15_output_path']) as im:
                old=np.array(im);diff=int(np.count_nonzero(q!=old))
            check={'source':p,'clamp_exact_equal':True,'finite':True,'historical_png_different_channels':diff,'forward_seconds':time.perf_counter()-t}
            checks.append(check)
            if diff:
                dump(EXP/'metrics/consistency_failure.json',checks)
                raise RuntimeError(p+' historical E15 PNG differs; stopping without analysis')
            dest=EXP/'outputs'/(Path(p).stem+'_reexport.png');Image.fromarray(q).save(dest)
            with Image.open(dest) as im: assert im.mode=='RGB' and np.array_equal(np.array(im),q)
            check['png_roundtrip_exact_equal']=True
            cache[p]=(a,za,ya,q)
            print('CONSISTENCY PASS',p,flush=True)
            del x,y,c,b,k,z
    assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
    dump(EXP/'metrics/consistency.json',checks)
    rows=[]
    for p,(a,z,y,q) in cache.items():
        x=a.astype(np.float32)/255.
        # Match frozen mask construction on CPU torch, avoiding reduction-order ambiguity.
        xt=torch.from_numpy(a/255.).float().permute(2,0,1)
        lum=(xt*torch.tensor([.2126,.7152,.0722])[:,None,None]).sum(0).numpy()
        masks={'all':np.ones(lum.shape,bool),'dark':lum<.2,'middle':(lum>=.2)&(lum<=.7),'bright':lum>.7}
        assert int(masks['dark'].sum())==fixed[p]['dark_pixel_count'] and int(masks['bright'].sum())==fixed[p]['bright_pixel_count']
        rows.append({'source':p,'regions':{name:summarize(x,z,y,q,mask) for name,mask in masks.items()}})
    summary=aggregate(rows)
    dump(EXP/'metrics/per_image.json',rows);dump(EXP/'metrics/summary.json',summary)
    roi_results=roi_analysis(rois,cache);dump(EXP/'metrics/roi_statistics.json',roi_results)
    for p,h in protected.items(): assert sha(ROOT/p)==h,p+' changed'
    dump(EXP/'metrics/verification.json',{'status':'PASS','images':8,'rois':12,'checkpoint_hash_verified':True,
        'strict_load':True,'weights_only':True,'exact_clamp':True,'historical_png_exact':True,'png_roundtrip_exact':True,
        'parameters_unchanged':True,'protected_files_unchanged':True,'nan_inf':False,'no_train_val_test_quarantine_used':True})
    lines=['# EXP006C：浮点越界与量化饱和归因分析','',
      '状态：PASS。仅冻结E15的8张dev推理与只读分析，无训练、网络/loss修改、checkpoint或split改写。', '',
      '## 数值一致性','',
      'SHA256：'+EXPECTED+'。weights_only=True、strict=True。全部8图clamp(z)与原模型输出torch.equal；重导出PNG与历史E15逐像素一致，保存重载一致。固定输入RGB哈希、区域计数及12个EXP006A ROI坐标沿用；源码、输入、权重和split结束后哈希未变。', '',
      '## 定义与分母','',
      '通道比例分母为区域像素数×3；any/all比例分母为区域像素数。input_existing指同一1280×720输入RGB的精确1端点，并非传感器RAW饱和。new_over_region分母为整个区域，new_over_input_nonendpoint分母为输入未达到对应channel/any/all端点的元素。all模式的新增全白可以来自输入已有单通道饱和，不能误称完全无输入饱和。', '',
      'channel端点分为z>1、z==1、0<z<1且round后255三类；另提供0<z<=1且PNG255的总非越界类。any端点独占归因按strict clip优先、exact1其次、纯rounding最后。该优先归因只说明存在越界通道，不能说明同像素所有端点均由越界产生。', '',
      '全白独占归因：存在strict clip / 无clip但有exact1 / 全部由rounding。white_rounding_assisted与clip可能重合：只要全白且不是所有浮点通道已为1，就需要量化参与。统计同时提供all z>1、all Y==1。', '',
      'pooled按实际分母合并，image_mean为非空图像比率等权均值；空区域null。亮度是编码RGB Yprime，不是线性物理亮度。', '',
      '## 全图逐图结果（百分比）','',
      '|图像|z>1通道|z>1任一像素|Y==1通道|PNG255通道|纯舍入通道|新增PNG任一像素/全图|全白像素|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    keys=['pre_gt1_channel','pre_gt1_any_pixel','post_eq1_channel','png255_channel','rounding_only_channel','png255_any_pixel_new_over_region','png255_all_channels_pixel']
    for row in rows:
        m=row['regions']['all']['metrics']
        lines.append('|'+Path(row['source']).stem+'|'+'|'.join(f"{100*m[k]['ratio']:.6f}" for k in keys)+'|')
    lines+=['','## 区域加权汇总（百分比）','',
       '|区域|像素数|z>1通道|z<0通道|Y==1通道|PNG255通道|纯舍入通道|新增PNG任一像素/区域|',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
    for region,s in summary.items():
        m=s['pooled'];kk=['pre_gt1_channel','pre_lt0_channel','post_eq1_channel','png255_channel','rounding_only_channel','png255_any_pixel_new_over_region']
        lines.append('|'+region+'|'+str(s['pixels'])+'|'+'|'.join(f"{100*m[k]['ratio']:.6f}" for k in kk)+'|')
    lines+=['','## 固定ROI与限制','',
      '0049/0179/0758/0792各3个ROI，未按输出重新选择。config/roi_coordinates.json保留720p坐标及×3原图定位（LANCZOS邻域意味着不是原生4K等价统计）。outputs/index.html展示输入/输出及固定中心行、中心列、全行/全列平均剖面。metrics/roi_statistics.json包含z/Y全范围直方图、分位数、亮度增益和水平/垂直/梯度幅值统计。outputs/*_floats.npz保留局部精确数组。', '',
      '梯度是相邻像素一阶差分，没有GT；噪声、曝光与截断均影响它。不同亮度下梯度减小或端点增加不能单独证明真实细节丢失。全白均匀区也可能是原本真实无纹理的灯光。', '',
      '## 实验判断','',
      '归因统计描述固定E15输出的数值机制，不证明训练loss是唯一成因，不证明H1-v3成立。若严格越界占端点主要部分，足以提出已有boundary机制的受控对照动机；不能由此保证它同时保持暗部可见性、提高质量或具有原创性。任何对照应另行批准，从随机初始化、同预算与冻结数据协议开始，联合审核暗区增益、新增饱和和结构证据。本次不启动对照。', '',
      '## 可追溯输出','',
      '- config/run.json：冻结哈希、输入和数值定义。',
      '- config/roi_coordinates.json：原EXP006A坐标。',
      '- metrics/consistency.json、verification.json：严格加载及一致性。',
      '- metrics/per_image.json、summary.json：逐图、分区、两种汇总。',
      '- metrics/roi_statistics.json：12个固定ROI的分布/剖面/梯度。',
      '- [图像与剖面索引](outputs/index.html)：仅输出目录，Git ignored。',
      '', '运行：E:\\anaconda3\\envs\\nightuav\\python.exe -B scripts/analyze_exp006c_float_saturation.py',
      '', '没有commit/push。完成后停止，等待人工审核。']
    (EXP/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('EXP006C PASS',flush=True)

if __name__=='__main__':
    try: main()
    except Exception as e:
        if EXP.exists():
            p=EXP/'failure.json'
            if not p.exists(): dump(p,{'status':'STOPPED','error':repr(e),'traceback':traceback.format_exc()})
        raise
