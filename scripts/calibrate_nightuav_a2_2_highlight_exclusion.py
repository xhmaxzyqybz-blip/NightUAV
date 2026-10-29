"""Locked A2-2 train-only calibration. No optimizer, parameter updates, or dev inference."""
from pathlib import Path
import hashlib,json,math,random,sys
import numpy as np
import torch
from PIL import Image
import train_di_retinex_exp005b as base
from prepare_nightuav_a2_2_highlight_exclusion import input_mask,structural_moments,structural_loss
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/nightuav_a2_2_highlight_exclusion'
OLD=ROOT/'experiments/nightuav_a2_1_structural_prior'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(name,v):
 with (OUT/name).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def norm(gs):return sum(float(g.double().square().sum()) for g in gs if g is not None)**.5
def moments(a,x):return structural_moments(a,x,preserve_a21_exclusions=True)
def loss(r,x,t):return structural_loss(r,x,t,preserve_a21_exclusions=True)
def main():
 assert not (OUT/'calibration_locked.json').exists() and not (OUT/'day_reference_locked.json').exists(),'Refusing overwrite/repeat'
 cfg=read(OLD/'config.json');oldcal=read(OLD/'struct_calibration.json');day=read(OLD/'day_reference.json')
 assert oldcal['rule'].startswith('lambda_struct=.1*RMS(||grad(rec+fixedTV)||)')
 protected={}
 for name in ['nightuav_a0_retinex','nightuav_a1_0_reconstruction','nightuav_a1_1_tv','nightuav_a2_1_structural_prior']:
  for p in (ROOT/'experiments'/name).rglob('*'):
   if p.is_file():protected[p]=base.sha(p)
 for p in [ROOT/'code/src/enhancement/nightuav_a0_retinex.py',ROOT/'scripts/train_nightuav_a2_1_structural_prior.py',ROOT/'scripts/prepare_nightuav_a2_2_highlight_exclusion.py',ROOT/'configs/splits/u3d_train_v2_1.txt']:
  protected[p]=base.sha(p)
 assert protected[ROOT/'scripts/train_nightuav_a2_1_structural_prior.py']==cfg['script_sha256']
 sums=torch.zeros(4,dtype=torch.float64);numbers=torch.zeros(4,dtype=torch.int64);records=[]
 with torch.no_grad():
  for row in day['rows']:
   p=ROOT/row['source'];assert p.resolve().is_relative_to((ROOT/'datasets/U3D/light/train').resolve()) and base.sha(p)==row['sha256']
   xx,yy,ww,hh=row['crop']['accepted_xywh'];assert ww==hh==256
   with Image.open(p) as im:
    assert im.size==(3840,2160);rgb=np.asarray(im.convert('RGB'))[yy:yy+hh,xx:xx+ww]
   x=torch.from_numpy(rgb/255.).float().permute(2,0,1)[None].contiguous().cuda()
   v,valid,c=moments(x,x);assert torch.isfinite(v).all()
   sums+=(v[0]*valid[0]).double().cpu();numbers+=valid[0].cpu()
   records.append(dict(source=row['source'],sha256=row['sha256'],crop=row['crop'],moments=v[0].tolist(),valid=valid[0].tolist(),counts=c[0].tolist()))
   if len(records)%100==0:print('REFERENCE',len(records),flush=True)
 assert len(records)==400 and (numbers>0).all()
 target=(sums/numbers).float().cuda().detach();target_before=target.clone()
 dump('day_reference_locked.json',dict(target=target.tolist(),valid_images=numbers.tolist(),rows=records,source_sha256=base.sha(OLD/'day_reference.json'),mask='.05<=luma<=.95 and maxRGB<.99 AND luma<=.7',sample_weight='equal valid-image per component; count0 excluded, count>=1 accepted'))
 # Import inherited loader before taking the RNG snapshot (module has a seed side effect).
 sys.path.insert(0,str(base.REPO));__import__('dataloader')
 sys.path.insert(0,str(ROOT/'code/src/enhancement'))
 from nightuav_a0_retinex import RetinexA0,reconstruction_loss
 random.seed(1149);np.random.seed(1149);torch.manual_seed(1149);torch.cuda.manual_seed_all(1149)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=RetinexA0().cuda()
 before={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
 initial_hash=hashlib.sha256(b''.join(t.numpy().tobytes() for t in before.values())).hexdigest()
 assert initial_hash==cfg['initial_parameter_sha256']
 py_rng=random.getstate();np_rng=np.random.get_state();cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all()
 train=(ROOT/'configs/splits/u3d_train_v2_1.txt').read_text().splitlines();assert len(train)==770
 ds=base.PatchDataset(train);params=tuple(model.parameters());rows=[]
 oldrows=[json.loads(v) for v in (OLD/'loss.jsonl').read_text().splitlines()]
 for j in range(8):
  items=[ds[tuple(v)] for v in cfg['plan'][2*j:2*j+2]]
  assert ds.records[-2:]==oldrows[j]['crops'] and [v[2] for v in items]==oldrows[j]['sources']
  x=torch.stack([v[0] for v in items]).cuda();o=model(x)
  y=(x*x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
  m=input_mask(x,preserve_a21_exclusions=True)
  assert torch.equal(m,((y>=.05)&(y<=.95)&(x.amax(1)<.99)&(y<=.7)))
  rec=reconstruction_loss(o['X_hat'],x)
  i=o['I'];tv=(i[:,:,1:,:]-i[:,:,:-1,:]).abs().mean()+(i[:,:,:,1:]-i[:,:,:,:-1]).abs().mean()
  basic=rec+cfg['lambda_tv']*tv;struct,coverage=loss(o['R'],x,target)
  assert all(torch.isfinite(t) for t in [rec,tv,basic,struct])
  assert all(coverage['valid_components']),'Empty calibration batch; stop without changing rules'
  gr=torch.autograd.grad(rec,params,retain_graph=True)
  gt=torch.autograd.grad(tv,params,allow_unused=True,retain_graph=True)
  gb=torch.autograd.grad(basic,params,retain_graph=True)
  output_grad=torch.autograd.grad(struct,o['R'],retain_graph=True)[0]
  excluded=(~m)[:,None].expand_as(output_grad);bright=(y>.7)[:,None].expand_as(output_grad)
  assert torch.isfinite(output_grad).all() and output_grad.abs().sum()>0
  assert (output_grad[excluded]==0).all() and (output_grad[bright]==0).all()
  gs=torch.autograd.grad(struct,params,allow_unused=True)
  assert all(g is None or torch.isfinite(g).all() for g in list(gr)+list(gt)+list(gb)+list(gs))
  groups={prefix:norm([g for (n,_),g in zip(model.named_parameters(),gs) if n.startswith(prefix)]) for prefix in ['encoder','reflectance_head','illumination_head']}
  assert groups['encoder']>1e-12 and groups['reflectance_head']>1e-12 and groups['illumination_head']==0
  nb,ns=norm(gb),norm(gs);assert norm(gr)>0 and norm(gt)>0 and ns>1e-12
  assert math.isclose(nb,oldcal['batches'][j]['base_grad_l2'],rel_tol=1e-7,abs_tol=1e-12)
  rows.append(dict(batch=j+1,sources=[v[2] for v in items],crops=ds.records[-2:],rec=float(rec.detach()),tv=float(tv.detach()),struct=float(struct.detach()),base=float(basic.detach()),base_grad_l2=nb,struct_grad_l2=ns,rec_grad_l2=norm(gr),tv_grad_l2=norm(gt),struct_group_grad_l2=groups,coverage=coverage,excluded_R_output_gradient_max=0.,bright_R_output_gradient_max=0.,bright_pixels=int((y>.7).sum()),empty_image_components=sum(not v for counts in coverage['valid_edges'] for v in counts)))
  print('CALIBRATED_BATCH',j+1,flush=True)
 base_rms=(sum(v['base_grad_l2']**2 for v in rows)/8)**.5
 struct_rms=(sum(v['struct_grad_l2']**2 for v in rows)/8)**.5
 weight=.1*base_rms/struct_rms;assert math.isfinite(weight)
 assert all(torch.equal(before[n],t.detach().cpu()) for n,t in model.state_dict().items())
 assert all(p.grad is None for p in model.parameters())
 assert random.getstate()==py_rng
 nr=np.random.get_state();assert nr[0]==np_rng[0] and np.array_equal(nr[1],np_rng[1]) and nr[2:]==np_rng[2:]
 assert torch.equal(cpu_rng,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu_rng,torch.cuda.get_rng_state_all()))
 assert torch.equal(target,target_before) and not target.requires_grad
 assert all(base.sha(p)==h for p,h in protected.items())
 result=dict(status='PASS_TRAIN_ONLY_CALIBRATION_NO_UPDATES',lambda_struct=weight,lambda_tv=cfg['lambda_tv'],base_gradient_rms=base_rms,struct_gradient_rms=struct_rms,target_ratio=.1,achieved_ratio=weight*struct_rms/base_rms,
  formula='0.1*sqrt(mean(batch_base_grad_l2^2))/sqrt(mean(batch_struct_grad_l2^2)); all10276 parameter vector; unused gradients zero',batches=rows,
  initial_parameter_sha256=initial_hash,parameters_and_buffers_unchanged=True,python_numpy_torch_cpu_cuda_rng_unchanged=True,target_fixed=True,historical_hashes_unchanged=True,parameter_updates=0,optimizer_created=False,dev_inference=False,
  empty_batches=0,exception=None,gradient_scope='Zero direct structural gradient at masked R output pixels; shared CNN weights can still affect highlight outputs after future updates')
 dump('calibration_locked.json',result)
 locked=dict(cfg);locked.update(status='CALIBRATED_NOT_TRAINED',preserve_a21_exclusions=True,lambda_struct=weight,mask='A2-1 (.05<=Y<=.95 AND maxRGB<.99) AND Y<=.7; both edge endpoints',day_reference='day_reference_locked.json',day_reference_sha256=base.sha(OUT/'day_reference_locked.json'),calibration='calibration_locked.json',training_enabled=False,script_sha256=base.sha(Path(__file__)),actual_changes=['additional upper-luma exclusion on both domains','reference recomputed using identical400 accepted crops','weight recalibrated to same10% initial gradient ratio'],source_config_sha256=base.sha(OLD/'config.json'))
 locked.pop('script_sha256',None);locked['calibration_script_sha256']=base.sha(Path(__file__))
 dump('config_locked.json',locked)
 print('PASS',json.dumps({k:v for k,v in result.items() if k!='batches'}),flush=True)
if __name__=='__main__':main()