"""Fixed B1 paired probe. --help is inert; --run requires separate human approval.
Resume only into a NEW experiment directory, from a completed-pair checkpoint.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('local31', 'global-image-mean')
E15 = ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth'
E15_HASH = '73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0'
SETTINGS = dict(seed=1149, max_attempts=500, beta=.63, delta_max=1., lr=.001,
    betas=[.9,.999], eps=1e-8, weight_decay=0., grad_clip=.1, batch_size=1,
    crop_size=256, dtype='float32', checkpoint_every=100, arms=list(ARMS),
    dark_threshold=.2, window=31, scheduler=None,
    zero_gradient_policy='skip optimizer.step; count attempt, not effective update',
    scale_note='31 pixels in native 4K crop training differs from 720p evaluation',
    global_condition='full original RGB training image mean; full 720p input mean at evaluation')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Execute the fixed approved protocol')
    parser.add_argument('--output-root', type=Path, required=True, help='New directory under project experiments')
    parser.add_argument('--resume', type=Path, help='Complete paired checkpoint; output directory must be new')
    return parser.parse_args()


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf8'))


def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def execute(args):
    # No torch/model/data imports or GPU access on --help or without --run.
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import train_di_retinex_exp005b as base
    from analyze_exp006c_float_saturation import summarize, aggregate, COEF
    sys.path.insert(0, str(ROOT/'code/src/enhancement'))
    from nightuav_v1_conditioned_di import ConditionedDI

    output = args.output_root.resolve()
    exp_root = (ROOT/'experiments').resolve()
    if output == exp_root or not output.is_relative_to(exp_root) or output.exists():
        raise ValueError('Output must be a NEW child of experiments; no overwriting')
    if args.resume and not args.resume.resolve().is_relative_to(exp_root):
        raise ValueError('Resume checkpoint must be inside project experiments')
    assert sha(E15) == E15_HASH
    _, official, split, meta = base.setup()
    official.set_seed(1149)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    old_cfg = read(base.EXP/'config/run.json')
    protected = {**old_cfg['protected_hashes'], **old_cfg['split_sha256']}
    protected['scripts/train_di_retinex_exp005b.py'] = old_cfg['script_sha256']
    protected[E15.relative_to(ROOT).as_posix()] = E15_HASH
    fixed = read(base.EXP/'config/fixed_dev_inputs.json')
    roi_path = ROOT/'experiments/exp006a_failure_analysis/config/crop_coordinates.json'
    rois = read(roi_path)
    assert len(fixed) == 8 and {r['source'] for r in fixed} == set(split['dev'])
    assert len(rois) == 12 and all(r['source'] in split['dev'] for r in rois)
    for p in [Path(__file__), ROOT/'code/src/enhancement/nightuav_v1_conditioned_di.py',
              ROOT/'scripts/analyze_exp006c_float_saturation.py', roi_path,
              base.EXP/'config/fixed_dev_inputs.json']:
        protected[p.relative_to(ROOT).as_posix()] = sha(p)
    diag = read(ROOT/'experiments/exp006c_float_saturation_attribution/config/run.json')
    for source in split['dev']:
        for folder in ('input', 'epoch015'):
            p = ROOT/'results/demos/di_retinex_exp005b'/folder/(Path(source).stem+'.png')
            key = p.relative_to(ROOT).as_posix()
            protected[key] = diag['protected_hashes'][key]
    for p,h in protected.items(): assert sha(ROOT/p) == h, p
    config = dict(SETTINGS, e15_sha256=E15_HASH, protected_hashes=protected,
        torch=str(torch.__version__), cuda=torch.version.cuda, python=sys.version,
        numpy=str(np.__version__), gpu=torch.cuda.get_device_name(0))
    model = official.mymodel.enhance_net_nopool().cuda().eval()
    e15 = torch.load(E15, map_location='cpu', weights_only=True)
    assert e15['epoch'] == 15
    model.load_state_dict(e15['model'], strict=True)
    frozen = {n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
    del e15
    torch.manual_seed(1149)
    wrapper = ConditionedDI(model, delta_max=1.).cuda().eval()
    heads = {ARMS[0]: wrapper.condition, ARMS[1]: copy.deepcopy(wrapper.condition)}
    assert all(sum(p.numel() for p in h.parameters()) == 25 for h in heads.values())
    assert all(torch.equal(a,b) for a,b in zip(heads[ARMS[0]].parameters(), heads[ARMS[1]].parameters()))
    optimizers = {a:torch.optim.Adam(h.parameters(), lr=.001, betas=(.9,.999), eps=1e-8, weight_decay=0.)
                  for a,h in heads.items()}
    sampler, crop_rng = random.Random(1149), random.Random(1149)
    order = list(range(len(split['train'])))
    sampler.shuffle(order)
    plan = [(i,crop_rng.randint(0,3584),crop_rng.randint(0,1904)) for i in order[:500]]
    dataset = base.PatchDataset(split['train'])
    records, start = [], 0
    effective = {a:0 for a in ARMS}
    initial_eval = None
    coef = torch.tensor([.2126,.7152,.0722], dtype=torch.float32, device='cuda')[None,:,None,None]

    def luma(x): return (x*coef).sum(1,keepdim=True)

    def check_frozen():
        assert not model.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(torch.equal(t.detach().cpu(), frozen[n]) for n,t in model.state_dict().items())

    def rng_state():
        n = np.random.get_state()
        return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
            cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
            sampler=sampler.getstate(), crop=crop_rng.getstate(), replacement=dataset.retry_rng.getstate())

    def restore_rng(r):
        random.setstate(r['python'])
        n=r['numpy']; np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
        torch.set_rng_state(r['cpu']); torch.cuda.set_rng_state_all(r['cuda'])
        sampler.setstate(r['sampler']); crop_rng.setstate(r['crop']); dataset.retry_rng.setstate(r['replacement'])

    def control_forward(x, original, full_mean):
        # original v/k were extracted under no_grad in the existing wrapper.
        # Do NOT disable autograd on the condition head or render path.
        s = torch.full_like(original['s'], float(full_mean))
        delta = torch.tanh(heads[ARMS[1]](s))
        b,z,y = wrapper.render(x,original['v'],original['k'],delta)
        return dict(b=b,z=z,y=y,delta=delta)

    if args.resume:
        saved = torch.load(args.resume.resolve(), map_location='cpu', weights_only=True)
        assert saved['format'] == 'nightuav_b1_paired_v1' and saved['pair_complete'] is True
        assert saved['config'] == config and saved['e15_sha256'] == E15_HASH
        assert saved['plan'] == plan and saved['train_paths'] == split['train']
        start = saved['completed_attempts']; assert 0 <= start <= 500
        records = saved['records']; assert len(records) == start
        effective = saved['effective_updates']
        for a in ARMS:
            assert sum(r['arms'][a]['effective_update'] for r in records) == effective[a]
            heads[a].load_state_dict(saved['heads'][a], strict=True)
            optimizers[a].load_state_dict(saved['optimizers'][a])
        dataset.records = saved['crop_records']; assert len(dataset.records) == start
        initial_eval = saved['initial_eval']
        restore_rng(saved['rng'])
    check_frozen()
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'config.json',dict(config,resume_from=str(args.resume.resolve()) if args.resume else None))

    def evaluate(stage):
        rng = rng_state()
        before = {a:{n:t.detach().cpu().clone() for n,t in h.state_dict().items()} for a,h in heads.items()}
        rows = {a:[] for a in ('E15',)+ARMS}
        panel_dir = output/'roi'/stage
        panel_dir.mkdir(parents=True,exist_ok=False)
        with torch.no_grad():
            for item in fixed:
                source=item['source'];stem=Path(source).stem
                with Image.open(ROOT/'results/demos/di_retinex_exp005b/input'/f'{stem}.png') as im:
                    rgb=np.array(im.convert('RGB'))
                assert rgb.shape==(720,1280,3)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==item['resized_rgb_sha256']
                x=torch.from_numpy(rgb/255.).float().permute(2,0,1)[None].contiguous().cuda()
                local=wrapper(x);control=control_forward(x,local,float(luma(x).mean()))
                z0=local['k']*(x+.5*(local['original_b']-1))+.5*(local['original_b']+1)
                preds={'E15':dict(y=local['original_y'],z=z0),ARMS[0]:local,ARMS[1]:control}
                xin=rgb.astype(np.float32)/255.;yi=(xin*COEF).sum(2)
                masks={'all':np.ones_like(yi,dtype=bool),'dark':yi<.2,'middle':(yi>=.2)&(yi<=.7),'bright':yi>.7}
                ref=(local['original_y'][0].permute(1,2,0).cpu().numpy()*COEF).sum(2)
                pngs={}
                for a,pred in preds.items():
                    assert all(bool(torch.isfinite(pred[t]).all()) for t in ('y','z'))
                    y=pred['y'][0].permute(1,2,0).cpu().numpy();z=pred['z'][0].permute(1,2,0).cpu().numpy()
                    q=np.rint(y*255).astype('uint8');pngs[a]=q;yo=(y*COEF).sum(2)
                    dark=masks['dark'];n=int(dark.sum())
                    rows[a].append(dict(source=source,regions={name:summarize(xin,z,y,q,m) for name,m in masks.items()},
                        upper_mean=float((np.maximum(z-1,0).astype(np.float64)**2).mean()),
                        pixels=int(yi.size),dark_pixels=n,global_mean=float(yo.mean()),global_delta_e15=float((yo-ref).mean()),
                        dark_gain=float((yo-yi)[dark].mean()) if n else None,
                        dark_delta_e15=float((yo-ref)[dark].mean()) if n else None))
                with Image.open(ROOT/'results/demos/di_retinex_exp005b/epoch015'/f'{stem}.png') as im:
                    assert np.array_equal(pngs['E15'],np.array(im.convert('RGB')))
                for roi in (r for r in rois if r['source']==source):
                    xx,yy,ww,hh=roi['xywh'];canvas=Image.new('RGB',(ww*4,hh+24),'#202020');draw=ImageDraw.Draw(canvas)
                    for j,(label,img) in enumerate([('input',rgb),('E15',pngs['E15']),('B1',pngs[ARMS[0]]),('control',pngs[ARMS[1]])]):
                        canvas.paste(Image.fromarray(img[yy:yy+hh,xx:xx+ww]),(j*ww,24));draw.text((j*ww+3,4),label,fill='white')
                    with (panel_dir/f'{stem}_{roi["category"]}.png').open('xb') as f:canvas.save(f,format='PNG')
        result={}
        for a,values in rows.items():
            summary={}
            for key,weight in [('upper_mean','pixels'),('global_mean','pixels'),('global_delta_e15','pixels'),('dark_gain','dark_pixels'),('dark_delta_e15','dark_pixels')]:
                valid=[r for r in values if r[key] is not None];denom=sum(r[weight] for r in valid)
                summary[key]=dict(pooled=sum(r[key]*r[weight] for r in valid)/denom if denom else None,
                                  image_mean=float(np.mean([r[key] for r in valid])) if valid else None)
            result[a]=dict(per_image=values,regions=aggregate(values),brightness_and_upper=summary)
        check_frozen()
        assert all(torch.equal(t.detach().cpu(),before[a][n]) for a,h in heads.items() for n,t in h.state_dict().items())
        restore_rng(rng)
        dump(output/f'evaluation_{stage}.json',result)
        return result

    def save_checkpoint(completed):
        check_frozen()
        for p,h in protected.items():assert sha(ROOT/p)==h,p
        payload=dict(format='nightuav_b1_paired_v1',pair_complete=True,config=config,e15_sha256=E15_HASH,
            completed_attempts=completed,effective_updates=effective.copy(),plan=plan,train_paths=split['train'],
            heads={a:{n:t.detach().cpu().clone() for n,t in h.state_dict().items()} for a,h in heads.items()},
            optimizers={a:o.state_dict() for a,o in optimizers.items()},rng=rng_state(),
            records=records,crop_records=dataset.records,initial_eval=initial_eval)
        folder=output/'checkpoints';folder.mkdir(exist_ok=True)
        temporary=folder/f'attempt{completed:03d}.partial';final=folder/f'attempt{completed:03d}.pth'
        if final.exists():raise FileExistsError(final)
        with temporary.open('xb') as f:torch.save(payload,f)
        loaded=torch.load(temporary,map_location='cpu',weights_only=True)
        assert loaded['pair_complete'] and loaded['completed_attempts']==completed
        assert loaded['config']==config and loaded['plan']==plan
        for a,h in heads.items():
            assert all(torch.equal(t.cpu(),loaded['heads'][a][n]) for n,t in h.state_dict().items())
        temporary.rename(final)

    try:
        if initial_eval is None:initial_eval=evaluate('initial')
        else:dump(output/'evaluation_initial.json',initial_eval)
        with (output/'attempts.jsonl').open('x',encoding='utf8') as log:
            for r in records:log.write(json.dumps(r,allow_nan=False)+'\n')
            for attempt in range(start,500):
                check_frozen()
                X,G,source,cx,cy=dataset[plan[attempt]]
                assert source==split['train'][plan[attempt][0]] and source not in split['dev'] and source not in split['quarantine']
                assert sha(ROOT/source)==meta['image_sha256'][source]
                with Image.open(ROOT/source) as im:
                    assert im.size==(3840,2160)
                    rgb=np.array(im.convert('RGB'))
                full_mean=float((rgb.mean(axis=(0,1),dtype=np.float64)/255.) @ np.array([.2126,.7152,.0722]))
                del rgb,G
                x=X[None].cuda();assert x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and bool(x.any())
                local=wrapper(x);control=control_forward(x,local,full_mean)
                reference=luma(local['original_y']).detach();dark=luma(x)<.2;count=int(dark.sum())
                row=dict(attempt=attempt+1,source=source,crop=dataset.records[-1],full_image_luma_mean=full_mean,
                         dark_pixels=count,empty_dark=count==0,arms={})
                before={a:[p.detach().clone() for p in h.parameters()] for a,h in heads.items()}
                # Validate BOTH arms before stepping either; never checkpoint a partial pair.
                for a,pred in [(ARMS[0],local),(ARMS[1],control)]:
                    assert all(bool(torch.isfinite(pred[t]).all()) for t in ('y','z','delta'))
                    upper=torch.relu(pred['z']-1).square().mean()
                    dark_loss=(dark*(luma(pred['y'])-reference).square()).sum()/max(count,1)
                    loss=upper+.63*dark_loss;assert bool(torch.isfinite(loss))
                    optimizers[a].zero_grad(set_to_none=True);loss.backward()
                    params=list(heads[a].parameters())
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
                    nonzero=any(bool(torch.count_nonzero(p.grad)) for p in params)
                    norm=torch.nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True)
                    row['arms'][a]=dict(upper=float(upper.detach()),dark=float(dark_loss.detach()),loss=float(loss.detach()),
                        nonzero_training_gradient=nonzero,grad_norm_preclip=float(norm),optimizer_step=False,
                        effective_update=False,parameter_change_l1=0.,effective_updates=effective[a])
                for a in ARMS:
                    r=row['arms'][a]
                    if r['nonzero_training_gradient']:
                        optimizers[a].step();r['optimizer_step']=True
                    params=list(heads[a].parameters());assert all(bool(torch.isfinite(p).all()) for p in params)
                    change=sum(float((p.detach().double()-old.double()).abs().sum()) for p,old in zip(params,before[a]))
                    assert math.isfinite(change)
                    r['parameter_change_l1']=change;r['effective_update']=change>0
                    effective[a]+=int(change>0);r['effective_updates']=effective[a]
                    optimizers[a].zero_grad(set_to_none=True)
                check_frozen()
                records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                del local,control,x,X,reference,dark,before,pred,loss,upper,dark_loss
                if (attempt+1)%100==0:
                    save_checkpoint(attempt+1)
                    print(json.dumps(dict(completed_attempts=attempt+1,effective_updates=effective)),flush=True)
        # If resuming the final completed pair, evaluate only; no new updates.
        if start==500:save_checkpoint(500)
        evaluate('final')
        assert len(records)==500
        check_frozen()
        for p,h in protected.items():assert sha(ROOT/p)==h,p
        dump(output/'summary.json',dict(status='COMPLETED_500_PAIRED_ATTEMPTS',attempts=500,
             effective_updates=effective,extended=False,best_checkpoint_selected=False))
    except BaseException as error:
        # Preserve last complete checkpoint, including on user interruption.
        dump(output/'failure.json',dict(status='STOPPED_BOTH_ARMS',error=repr(error),traceback=traceback.format_exc(),
             completed_pairs=len(records),resume_rule='Complete paired .pth only, new output directory, explicit authorization'))
        raise


def main():
    args=arguments()
    if not args.run:
        raise SystemExit('No execution: --run requires explicit approval. Use --help to inspect the interface.')
    execute(args)


if __name__=='__main__':main()