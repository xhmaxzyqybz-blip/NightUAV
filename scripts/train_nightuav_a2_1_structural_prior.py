"""A2-1: one 200-update fixed daytime structural-prior run, paired to A1-1."""
import argparse,hashlib,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/nightuav_a2_1_structural_prior'
BASE=ROOT/'experiments/nightuav_a1_1_tv'

def structural_moments(a, x):
 import torch
 weights=a.new_tensor([.2126,.7152,.0722])[None,:,None,None]
 y=(a*weights).sum(1); xin=(x.detach()*weights).sum(1)
 mask=(xin>=.05)&(xin<=.95)&(x.detach().amax(1)<.99)
 logy=torch.log(y+1/255.)
 values=[];valid=[];counts=[]
 for dim in [-1,-2]:
  q=torch.tanh(torch.diff(logy,dim=dim).abs())
  m=(mask[...,1:]&mask[...,:-1]) if dim==-1 else (mask[:,1:,:]&mask[:,:-1,:])
  count=m.sum((-2,-1));ok=count>0
  for power in [1,2]:
   values.append((q.pow(power)*m).sum((-2,-1))/count.clamp_min(1))
   valid.append(ok);counts.append(count)
 return torch.stack(values,1),torch.stack(valid,1),torch.stack(counts,1)

def structural_loss(r,x,target):
 import torch
 values,valid,counts=structural_moments(r,x)
 denom=valid.sum(0);ok=denom>0
 mean=(values*valid).sum(0)/denom.clamp_min(1)
 loss=((mean-target.detach()).square()*ok).sum()/ok.sum().clamp_min(1)
 return loss,dict(moments=mean.detach().tolist(),valid_components=ok.tolist(),valid_edges=counts.detach().tolist())

def extra_diagnostics(o,x,target):
 import torch
 loss,info=structural_loss(o['R'],x,target)
 err=(o['X_hat']-x).abs();y=(x*x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
 regions={}
 for name,m in [('dark',y<.2),('middle',(y>=.2)&(y<=.7)),('bright',y>.7)]:
  regions[name]=dict(pixels=int(m.sum()),mae=float(err.permute(0,2,3,1)[m].mean()) if m.any() else None)
 def highpass(a):
  y=(a*a.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
  c=y[:,1:-1,1:-1]
  lap=(4*c-y[:,:-2,1:-1]-y[:,2:,1:-1]-y[:,1:-1,:-2]-y[:,1:-1,2:]).abs()
  return lap
 # Fixed input low-gradient interiors: descriptive high-frequency proxy, not noise GT.
 xy=y[:,1:-1,1:-1]
 flat=(y[:,1:-1,2:]-xy).abs()<.01
 flat &= (y[:,1:-1,:-2]-xy).abs()<.01
 flat &= (y[:,2:,1:-1]-xy).abs()<.01
 flat &= (y[:,:-2,1:-1]-xy).abs()<.01
 lap=highpass(o['R'])
 return dict(rgb_mae=err.mean((0,2,3)).tolist(),regions=regions,structure_distance=float(loss),structure=info,
  R_luma_abs_laplacian=float(lap.mean()),input_flat_pixels=int(flat.sum()),
  R_luma_abs_laplacian_input_flat=float(lap[flat].mean()) if flat.any() else None,
  high_frequency_note='4-neighbor luminance Laplacian absolute mean; fixed input four-neighbor differences <.01; not a noise measurement or quality score')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');a=p.parse_args()
 if not a.run:raise SystemExit('Use --run for the authorized single run')
 if EXP.exists():raise FileExistsError('Refuse overwrite or automatic retry: '+str(EXP))
 import torch,numpy as np
 from PIL import Image,ImageDraw
 import train_di_retinex_exp005b as base
 sys.path.insert(0,str(ROOT/'code/src/enhancement'))
 from nightuav_a0_retinex import RetinexA0,reconstruction_loss
 read=lambda p:json.loads(p.read_text(encoding='utf8'))
 def dump(p,v):
  with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
 protected=read(ROOT/'experiments/nightuav_a0_retinex/numeric_checks.json')['protected_sha256']
 protected.update(read(ROOT/'experiments/nightuav_a0_retinex/numeric_checks.json')['source_sha256'])
 for path,h in protected.items():assert base.sha(ROOT/path)==h,path
 baseline_cfg=read(BASE/'config.json')
 assert base.sha(ROOT/'scripts/train_nightuav_a1_1_tv.py')==baseline_cfg['script_sha256']
 old_rows=[json.loads(line) for line in (BASE/'loss.jsonl').read_text().splitlines()]
 assert len(old_rows)==200 and read(BASE/'summary.json')['effective_updates']==200
 for name in ['config.json','loss.jsonl','final.pth','summary.json','diagnostics_final.json','initial/0049_raw.npz','initial/0179_raw.npz','final/0049_raw.npz','final/0179_raw.npz']:
  path=BASE/name;protected[str(path.relative_to(ROOT))]=base.sha(path)
 _,official,split,meta=base.setup();official.set_seed(1149)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=RetinexA0().cuda();assert sum(p.numel() for p in model.parameters())==10276
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8)
 sampler=random.Random(1149);crop_rng=random.Random(1149);order=list(range(770));sampler.shuffle(order)
 plan=[(i,crop_rng.randint(0,3584),crop_rng.randint(0,1904)) for i in order[:400]]
 assert [list(v) for v in plan]==baseline_cfg['plan'], 'Sampling plan mismatch; stop'
 plan=[tuple(v) for v in baseline_cfg['plan']]
 ds=base.PatchDataset(split['train'])
 # Reconstruct the original initializer deterministically, without final weights.
 initial_parameters={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
 torch.manual_seed(1149);torch.cuda.manual_seed_all(1149)
 reference_initial=RetinexA0().cuda()
 assert all(torch.equal(t.detach().cpu(),initial_parameters[n]) for n,t in reference_initial.state_dict().items())
 del reference_initial
 torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
 initial_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in initial_parameters.values())).hexdigest()
 assert initial_hash==baseline_cfg['initial_parameter_sha256'], 'Initial parameters differ from A1-1'
 def tv_loss(i):
  return (i[:,:,1:,:]-i[:,:,:-1,:]).abs().mean()+(i[:,:,:,1:]-i[:,:,:,:-1]).abs().mean()
 # Fixed A1-1 weight: user decimal is a display rounding, never recalibrated.
 lambda_tv=baseline_cfg['lambda_tv']
 EXP.mkdir()
 from nightuav_crop_adapter import select_crop
 day_paths=sorted((ROOT/'datasets/U3D/light/train').glob('*.jpg'))
 assert len(day_paths)==800
 dr=random.Random(1149);dc=random.Random(1149);retry=random.Random(1149)
 indices=list(range(800));dr.shuffle(indices)
 day_plan=[(idx,dc.randint(0,3584),dc.randint(0,1904)) for idx in indices[:400]]
 dump(EXP/'day_plan.json',dict(seed=1149,plan=[dict(source=day_paths[i].relative_to(ROOT).as_posix(),xy=[x,y]) for i,x,y in day_plan]))
 reference_rows=[];sums=torch.zeros(4,dtype=torch.float64);number=torch.zeros(4,dtype=torch.int64)
 # Only byte-duplicate leakage is checked here; no new scene similarity audit.
 dev_hashes={base.sha(ROOT/v) for v in split['dev']}
 with torch.no_grad():
  for idx,xx,yy in day_plan:
   path=day_paths[idx];digest=base.sha(path);assert digest not in dev_hashes,'Day/dev identical bytes; stop'
   with Image.open(path) as im:
    assert im.size==(3840,2160);rgb=np.asarray(im.convert('RGB'))
   xx,yy,record=select_crop(rgb.transpose(2,0,1),path.relative_to(ROOT).as_posix(),retry,(xx,yy),size=256,max_retries=50)
   x=torch.from_numpy(rgb[yy:yy+256,xx:xx+256]/255.).float().permute(2,0,1)[None].contiguous().cuda()
   values,valid,counts=structural_moments(x,x)
   assert torch.isfinite(values).all()
   sums+=(values[0]*valid[0]).double().cpu();number+=valid[0].cpu()
   reference_rows.append(dict(source=path.relative_to(ROOT).as_posix(),sha256=digest,crop=record,moments=values[0].tolist(),valid=valid[0].tolist(),valid_edges=counts[0].tolist()))
   if len(reference_rows)%50==0:print('DAY_REFERENCE',len(reference_rows),flush=True)
 assert bool((number>0).all()),'No valid daytime target component'
 target=(sums/number).float().cuda().detach();target_before=target.clone()
 assert not target.requires_grad and target.grad_fn is None
 reference_record=dict(seed=1149,crops=400,target=target.tolist(),valid_images_per_component=number.tolist(),
  component_order=['horizontal_mean','horizontal_second_raw_moment','vertical_mean','vertical_second_raw_moment'],
  expression='q=tanh(abs(diff(log(.2126R+.7152G+.0722B+1/255)))); one-pixel adjacency at native 4K',
  mask='Both input endpoints: .05<=luma<=.95 and maxRGB<.99; fixed, detached',
  normalization='Valid-edge mean per image, then equal valid-image mean per component; second moment is mean(q^2), not variance',
  preprocessing='Pillow RGB, native crop, divide255 in numpy float64 then cast FP32, identical RGB normalization to inherited loader; no Gaussian used',
  scene_overlap='No established cross-domain scene audit; checked exact-byte overlap with dev only; exploratory not independent generalization',rows=reference_rows)
 dump(EXP/'day_reference.json',reference_record)
 # One initial train-only calibration. No parameter updates or RNG consumption.
 cal_ds=base.PatchDataset(split['train']);calibration=[];params=tuple(model.parameters())
 for j in range(8):
  items=[cal_ds[v] for v in plan[2*j:2*j+2]]
  assert cal_ds.records[-2:]==old_rows[j]['crops'] and [v[2] for v in items]==old_rows[j]['sources']
  x=torch.stack([v[0] for v in items]).cuda();o=model(x)
  rec=reconstruction_loss(o['X_hat'],x);tv=tv_loss(o['I']);basic=rec+lambda_tv*tv
  struct,coverage=structural_loss(o['R'],x,target)
  assert torch.isfinite(struct) and all(coverage['valid_components']), 'Invalid calibration coverage'
  if j==0:assert float(rec.detach())==read(BASE/'summary.json')['initial_fixed_train_batch_loss']
  gb=torch.autograd.grad(basic,params,retain_graph=True)
  gs=torch.autograd.grad(struct,params,allow_unused=True)
  assert all(torch.isfinite(v).all() for v in gb) and all(v is None or torch.isfinite(v).all() for v in gs)
  groups={}
  for prefix in ['encoder','illumination_head','reflectance_head']:
   groups[prefix]=sum(float(g.double().square().sum()) for (name,_),g in zip(model.named_parameters(),gs) if name.startswith(prefix) and g is not None)**.5
  assert groups['encoder']>1e-12 and groups['reflectance_head']>1e-12 and groups['illumination_head']==0, 'Structural path failure'
  nb=sum(float(v.double().square().sum()) for v in gb)**.5
  ns=sum(float(v.double().square().sum()) for v in gs if v is not None)**.5
  assert ns>1e-12, 'Near-zero structural gradient; no weight amplification'
  calibration.append(dict(batch=j+1,rec=float(rec.detach()),tv=float(tv.detach()),basic=float(basic.detach()),struct=float(struct.detach()),base_grad_l2=nb,struct_grad_l2=ns,struct_group_grad_l2=groups,coverage=coverage,sources=[v[2] for v in items]))
  del x,o,rec,tv,basic,struct,gb,gs,items
 base_rms=(sum(v['base_grad_l2']**2 for v in calibration)/8)**.5
 struct_rms=(sum(v['struct_grad_l2']**2 for v in calibration)/8)**.5
 lambda_struct=.1*base_rms/struct_rms
 assert __import__('math').isfinite(lambda_struct) and lambda_struct>0
 assert all(torch.equal(t.detach().cpu(),initial_parameters[n]) for n,t in model.state_dict().items())
 assert torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all()))
 calibration_record=dict(rule='lambda_struct=.1*RMS(||grad(rec+fixedTV)||)/RMS(||grad(struct)||), first8 paired train batches, all parameters; unused I-head gradient zero',
  lambda_struct=lambda_struct,lambda_tv=lambda_tv,base_gradient_rms=base_rms,struct_gradient_rms=struct_rms,batches=calibration,
  minimum_calibration_coverage='Each batch has all four valid components and finite encoder/R-head gradients >1e-12; not a physical/statistical coverage guarantee',
  runtime_check='PASS; finite loss, R-head/encoder gradients, no I-head direct gradient, target requires_grad=False; no update',
  reconstruction_acceptance='No numerical tolerance approved; report all changes without declaring quality acceptance',
  note='Single preset 10% initial gradient ratio; no optimality or stable future dynamics guarantee')
 dump(EXP/'struct_calibration.json',calibration_record)
 def overlap(i,r):
  a=np.concatenate([np.abs(np.diff(i.astype(np.float64),axis=0)).ravel(),np.abs(np.diff(i.astype(np.float64),axis=1)).ravel()])
  b=np.concatenate([np.abs(np.diff(r.astype(np.float64),axis=0)).mean(2).ravel(),np.abs(np.diff(r.astype(np.float64),axis=1)).mean(2).ravel()])
  den=np.linalg.norm(a)*np.linalg.norm(b)
  return dict(magnitude_cosine=float(np.dot(a,b)/den) if den else None,I_mean_abs_adjacent=float(a.mean()),R_mean_abs_adjacent=float(b.mean()),
   definition='Cosine of concatenated absolute horizontal/vertical differences; R averaged over RGB; no edge threshold; not physical correctness')
 cfg=dict(seed=1149,updates=200,batch_size=2,crop=256,lr=.001,weight_decay=0,betas=[.9,.999],eps=1e-8,
  loss='L1 reconstruction + fixed lambda_tv * TV(I) + fixed lambda_struct * structure',lambda_struct=lambda_struct,day_reference_sha256=base.sha(EXP/'day_reference.json'),lambda_tv=lambda_tv,initial_parameter_sha256=initial_hash,gradient_clipping=None,scheduler=None,precision='FP32',plan=plan,
  data='Frozen train v2.1; reuse PatchDataset then explicitly stack 2 samples; full-image Gaussian computed by inherited loader but unused',
  crop_protocol='Strict all-zero only; max50 same-image replacements; no sample substitution',
  evaluation='0049/0179 fixed saved 1280x720 inputs; before/after only; no gradients or RNG updates',
  visualization='Fixed [0,1] to np.rint(value*255) RGB PNG; no per-image minmax, exposure or sharpening; I repeated as gray',
  diagnostic_thresholds={'near_boundary':[.01,.99],'near_constant_I_spatial_std':.01,'note':'descriptive, not tuned and not physical validation'},
  initialization='A0 random initializer; no checkpoint loaded',torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
  protected_hashes=protected,script_sha256=base.sha(Path(__file__)))
 dump(EXP/'config.json',cfg)
 print('RUNTIME_CHECK_PASS_FIXED_WEIGHTS',lambda_tv,lambda_struct,flush=True)
 updates=0;rows=[];first_batch=None;start=time.perf_counter()
 fixed={Path(v['source']).stem:v for v in read(ROOT/'experiments/exp005b_di_retinex_u3d_baseline/config/fixed_dev_inputs.json')}
 def distribution(t):
  z=t.detach().double().flatten();q=torch.quantile(z,z.new_tensor([0,.01,.1,.5,.9,.99,1]))
  return dict(mean=float(z.mean()),std=float(z.std(unbiased=False)),quantile_levels=[0,.01,.1,.5,.9,.99,1],quantiles=q.tolist(),
   fraction_le_001=float((z<=.01).double().mean()),fraction_ge_099=float((z>=.99).double().mean()),
   exact_zero=float((z==0).double().mean()),exact_one=float((z==1).double().mean()))
 def evaluate(stage):
  before={n:t.detach().clone() for n,t in model.state_dict().items()};rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all();model.eval();results=[]
  folder=EXP/stage;folder.mkdir()
  with torch.no_grad():
   for stem in ['0049','0179']:
    assert fixed[stem]['source'] in split['dev']
    with Image.open(ROOT/'results/demos/di_retinex_exp005b/input'/f'{stem}.png') as im:rgb=np.array(im.convert('RGB'))
    assert rgb.shape==(720,1280,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==fixed[stem]['resized_rgb_sha256']
    x=torch.from_numpy(rgb/255.).float().permute(2,0,1)[None].contiguous().cuda();o=model(x)
    assert all(bool(torch.isfinite(t).all()) for t in o.values())
    i=o['I'];r=o['R'];ix=distribution(i);rx=distribution(r)
    spatial_std=float(i.std(dim=(-2,-1),unbiased=False).mean())
    tv=float((i[:,:,:,1:]-i[:,:,:,:-1]).abs().mean()+(i[:,:,1:,:]-i[:,:,:-1,:]).abs().mean())
    results.append(dict(image=stem,reconstruction_mae=float(reconstruction_loss(o['X_hat'],x)),I=ix,R=rx,
     R_channels={str(c):distribution(r[:,c:c+1]) for c in range(3)},I_spatial_std=spatial_std,I_TV_mean_adjacent_differences=tv,
     near_constant_I_std_below_001=spatial_std<.01,I_distance_from_one=float((1-i).abs().mean()),R_distance_from_X=float((r-x).abs().mean())))
    arrays={k:t[0].permute(1,2,0).cpu().numpy() for k,t in o.items()}
    results[-1]['gradient_overlap']=overlap(arrays['I'],arrays['R'])
    results[-1]['extra']=extra_diagnostics(o,x,target)
    if stage=='initial':
     with np.load(BASE/'initial'/f'{stem}_raw.npz') as prior:
      assert all(np.array_equal(arrays[k],prior[k]) for k in arrays),'Initial prediction mismatch; no training'
     results[-1]['historical_initial_float_exact']=True
    with (folder/f'{stem}_raw.npz').open('xb') as f:np.savez_compressed(f,**arrays)
    canvas=Image.new('RGB',(5120,744),'#202020');draw=ImageDraw.Draw(canvas)
    for j,(name,img) in enumerate([('X',rgb),('I',np.repeat(np.rint(arrays['I']*255).astype('uint8'),3,axis=2)),
      ('R',np.rint(arrays['R']*255).astype('uint8')),('X_hat',np.rint(arrays['X_hat']*255).astype('uint8'))]):
     canvas.paste(Image.fromarray(img),(j*1280,24));draw.text((j*1280+8,5),stem+' '+stage+' '+name,fill='white')
    with (folder/f'{stem}_decomposition.png').open('xb') as f:canvas.save(f,format='PNG')
  assert all(torch.equal(t,before[n]) for n,t in model.state_dict().items())
  assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(crng,torch.cuda.get_rng_state_all()))
  model.train();dump(EXP/f'diagnostics_{stage}.json',results);return results
 def batch(attempt):
  items=[ds[p] for p in plan[attempt*2:attempt*2+2]]
  for item in items:assert item[2] in split['train'] and item[2] not in split['dev'] and item[2] not in split['quarantine']
  x=torch.stack([v[0] for v in items]).cuda()
  assert x.shape==(2,3,256,256) and bool(torch.isfinite(x).all()) and all(bool(v.any()) for v in x)
  assert ds.records[-2:]==old_rows[attempt]['crops'] and [v[2] for v in items]==old_rows[attempt]['sources'],'Pairing mismatch; stop'
  return x,[v[2] for v in items]
 try:
  initial=evaluate('initial')
  with (EXP/'loss.jsonl').open('x',encoding='utf8') as log:
   for attempt in range(200):
    x,paths=batch(attempt)
    if attempt==0:first_batch=x.detach().cpu().clone()
    o=model(x);assert all(bool(torch.isfinite(v).all()) for v in o.values())
    rec=reconstruction_loss(o['X_hat'],x);tv=tv_loss(o['I']);struct,coverage=structural_loss(o['R'],x,target);loss=rec+lambda_tv*tv+lambda_struct*struct;assert bool(torch.isfinite(loss))
    assert torch.equal(target,target_before) and not target.requires_grad
    if attempt==0:
     initial_fixed=float(rec.detach());dump(EXP/'initial_state.json',dict(effective_updates=0,fixed_train_batch_loss=initial_fixed,sources=paths,crops=ds.records[:2]))
     print('INITIAL_FIXED_TRAIN_LOSS',initial_fixed,flush=True)
    optimizer.zero_grad(set_to_none=True);loss.backward()
    params=list(model.parameters());assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    grad_norm=sum(float(p.grad.detach().double().square().sum()) for p in params)**.5
    assert grad_norm>0,'Zero training gradient; stop without extending budget'
    before=[p.detach().clone() for p in params];optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in params)
    delta=sum(float((p.detach().double()-old.double()).abs().sum()) for p,old in zip(params,before))
    assert delta>0 and __import__('math').isfinite(delta),'Ineffective update; stop without replacement attempts'
    updates+=1
    row=dict(attempt=attempt+1,effective_updates=updates,loss_pre_update=float(loss.detach()),rec_pre_update=float(rec.detach()),tv_pre_update=float(tv.detach()),weighted_tv=float((lambda_tv*tv).detach()),grad_l2=grad_norm,
     struct_pre_update=float(struct.detach()),weighted_struct=float((lambda_struct*struct).detach()),structure=coverage,parameter_change_l1=delta,sources=paths,crops=ds.records[-2:])
    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    del x,o,loss,rec,tv,struct,before
    if updates%20==0:print(json.dumps(dict(effective_updates=updates,last_loss=row['loss_pre_update'],mean_last20=sum(v['loss_pre_update'] for v in rows[-20:])/20)),flush=True)
  optimizer.zero_grad(set_to_none=True)
  with torch.no_grad():
   x=first_batch.cuda();fixed_out=model(x);final_fixed=float(reconstruction_loss(fixed_out['X_hat'],x));final_fixed_structure=float(structural_loss(fixed_out['R'],x,target)[0]);del x,fixed_out
  final=evaluate('final')
  baseline_diagnostics={r['image']:r for r in read(BASE/'diagnostics_final.json')};comparison=[]
  for cur in final:
   stem=cur['image']
   with np.load(BASE/'final'/f'{stem}_raw.npz') as prior:
    old_overlap=overlap(prior['I'],prior['R'])
    with Image.open(ROOT/'results/demos/di_retinex_exp005b/input'/f'{stem}.png') as im:rgb=np.array(im.convert('RGB'))
    xt=torch.from_numpy(rgb/255.).float().permute(2,0,1)[None].cuda()
    ot={k:torch.from_numpy(prior[k]).permute(2,0,1)[None].cuda() for k in ['I','R','X_hat']}
    with torch.no_grad():old_extra=extra_diagnostics(ot,xt,target)
   comparison.append(dict(image=stem,A1_1=baseline_diagnostics[stem],A1_1_extra=old_extra,A2_1=cur))
  dump(EXP/'comparison.json',comparison)
  n=np.random.get_state()
  payload=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),
   config=cfg,seed=1149,effective_updates=updates,completed_attempts=len(rows),plan=plan,crop_records=ds.records,
   python_rng=random.getstate(),numpy_rng=[n[0],n[1].tolist(),n[2],n[3],n[4]],torch_cpu_rng=torch.get_rng_state(),
   torch_cuda_rng=torch.cuda.get_rng_state_all(),sampler_rng=sampler.getstate(),crop_rng=crop_rng.getstate(),replacement_rng=ds.retry_rng.getstate())
  with (EXP/'final.pth').open('xb') as f:torch.save(payload,f)
  restored=torch.load(EXP/'final.pth',map_location='cpu',weights_only=True)
  assert restored['effective_updates']==200 and all(torch.equal(t.cpu(),restored['model'][k]) for k,t in model.state_dict().items())
  for path,h in protected.items():assert base.sha(ROOT/path)==h,path
  dump(EXP/'summary.json',dict(status='COMPLETED_200_EFFECTIVE_UPDATES',effective_updates=updates,attempts=len(rows),
   initial_fixed_train_batch_loss=initial_fixed,final_fixed_train_batch_loss=final_fixed,
   first20_train_mean=sum(v['loss_pre_update'] for v in rows[:20])/20,last20_train_mean=sum(v['loss_pre_update'] for v in rows[-20:])/20,
   last_step_pre_update_loss=rows[-1]['loss_pre_update'],elapsed_seconds=time.perf_counter()-start,
   lambda_tv=lambda_tv,lambda_struct=lambda_struct,initial_fixed_structure=calibration[0]['struct'],final_fixed_structure=final_fixed_structure,target_unchanged=bool(torch.equal(target,target_before)),initialization_replay_parameter_exact=True,historical_initial_predictions_exact=True,paired_crop_records_exact=True,
   initial_parameter_sha256=initial_hash,last20_rec_mean=sum(v['rec_pre_update'] for v in rows[-20:])/20,checkpoint_reload_exact=True,protected_hashes_unchanged=True,exceptions=[],diagnostic_images=['0049','0179'],
   interpretation='One Cday run; no Cnight source control; matching structure statistics is not physical identifiability or independent generalization'))
  print('COMPLETE',json.dumps(read(EXP/'summary.json')),flush=True)
 except BaseException as error:
  dump(EXP/'failure.json',dict(status='STOPPED_NO_AUTOMATIC_RETRY',error=repr(error),traceback=traceback.format_exc(),
   effective_updates=updates,completed_logged_updates=len(rows)))
  raise

if __name__=='__main__':main()