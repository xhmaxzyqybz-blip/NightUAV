"""A1-0: one fixed 200-update reconstruction-only run, no continuation or tuning."""
import argparse,hashlib,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/nightuav_a1_0_reconstruction'

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
 _,official,split,meta=base.setup();official.set_seed(1149)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=RetinexA0().cuda();assert sum(p.numel() for p in model.parameters())==10276
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0,betas=(.9,.999),eps=1e-8)
 sampler=random.Random(1149);crop_rng=random.Random(1149);order=list(range(770));sampler.shuffle(order)
 plan=[(i,crop_rng.randint(0,3584),crop_rng.randint(0,1904)) for i in order[:400]]
 ds=base.PatchDataset(split['train'])
 cfg=dict(seed=1149,updates=200,batch_size=2,crop=256,lr=.001,weight_decay=0,betas=[.9,.999],eps=1e-8,
  loss='mean(abs(I*R-X)) only',gradient_clipping=None,scheduler=None,precision='FP32',plan=plan,
  data='Frozen train v2.1; reuse PatchDataset then explicitly stack 2 samples; full-image Gaussian computed by inherited loader but unused',
  crop_protocol='Strict all-zero only; max50 same-image replacements; no sample substitution',
  evaluation='0049/0179 fixed saved 1280x720 inputs; before/after only; no gradients or RNG updates',
  visualization='Fixed [0,1] to np.rint(value*255) RGB PNG; no per-image minmax, exposure or sharpening; I repeated as gray',
  diagnostic_thresholds={'near_boundary':[.01,.99],'near_constant_I_spatial_std':.01,'note':'descriptive, not tuned and not physical validation'},
  initialization='A0 random initializer; no checkpoint loaded',torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0),
  protected_hashes=protected,script_sha256=base.sha(Path(__file__)))
 EXP.mkdir();dump(EXP/'config.json',cfg)
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
  return x,[v[2] for v in items]
 try:
  initial=evaluate('initial')
  with (EXP/'loss.jsonl').open('x',encoding='utf8') as log:
   for attempt in range(200):
    x,paths=batch(attempt)
    if attempt==0:first_batch=x.detach().cpu().clone()
    o=model(x);assert all(bool(torch.isfinite(v).all()) for v in o.values())
    loss=reconstruction_loss(o['X_hat'],x);assert bool(torch.isfinite(loss))
    if attempt==0:
     initial_fixed=float(loss.detach());dump(EXP/'initial_state.json',dict(effective_updates=0,fixed_train_batch_loss=initial_fixed,sources=paths,crops=ds.records[:2]))
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
    row=dict(attempt=attempt+1,effective_updates=updates,loss_pre_update=float(loss.detach()),grad_l2=grad_norm,
     parameter_change_l1=delta,sources=paths,crops=ds.records[-2:])
    rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
    del x,o,loss,before
    if updates%20==0:print(json.dumps(dict(effective_updates=updates,last_loss=row['loss_pre_update'],mean_last20=sum(v['loss_pre_update'] for v in rows[-20:])/20)),flush=True)
  optimizer.zero_grad(set_to_none=True)
  with torch.no_grad():
   x=first_batch.cuda();final_fixed=float(reconstruction_loss(model(x)['X_hat'],x));del x
  final=evaluate('final')
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
   checkpoint_reload_exact=True,protected_hashes_unchanged=True,exceptions=[],diagnostic_images=['0049','0179'],
   interpretation='Reconstruction optimization only; no claim of physical identifiability or enhancement quality'))
  print('COMPLETE',json.dumps(read(EXP/'summary.json')),flush=True)
 except BaseException as error:
  dump(EXP/'failure.json',dict(status='STOPPED_NO_AUTOMATIC_RETRY',error=repr(error),traceback=traceback.format_exc(),
   effective_updates=updates,completed_logged_updates=len(rows)))
  raise

if __name__=='__main__':main()