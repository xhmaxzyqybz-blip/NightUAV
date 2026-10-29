"""Approved EXP007A-0 200-update smoke; no dev, checkpoint load or continuation."""
import json,random,time,math,statistics,traceback,hashlib
from pathlib import Path
import torch
import train_di_retinex_exp005b as base
from nightuav_boundary_adapter import forward_with_original_z,upper_loss
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/exp007a0_boundary_mechanism_check'
LAM=0.053192434320470425
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs))
def stats(v):
 return {'mean':statistics.mean(v),'min':min(v),'max':max(v),'median':statistics.median(v),'rms':math.sqrt(statistics.mean(x*x for x in v))}
def main():
 assert not(EXP/'config/smoke_run.json').exists(),'No overwrite/retry'
 check=read(EXP/'config/run.json');assert read(EXP/'metrics/verification.json')['gradient_checks']=='PASS'
 assert base.sha(ROOT/'scripts/nightuav_boundary_adapter.py')==check['adapter_sha256']
 assert base.sha(ROOT/'scripts/check_exp007a0_boundary.py')==check['script_sha256']
 for p,h in check['protected_hashes'].items():assert base.sha(ROOT/p)==h,p
 torch_,official,split,meta=base.setup()
 official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train()
 init=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 baseline_cfg=read(ROOT/'experiments/exp005b_di_retinex_u3d_baseline/config/run.json')
 assert init==baseline_cfg['initial_model_sha256'],'Fresh initialization differs'
 params=tuple(model.parameters());optimizer=torch.optim.Adam(params,lr=.001,weight_decay=.0001)
 fn,expressions=base.make_loss(torch)
 order=list(range(770));random.Random(1149).shuffle(order);rng=random.Random(1149)
 plan=[(idx,rng.randint(0,3584),rng.randint(0,1904)) for idx in order][:200]
 ds=base.PatchDataset(split['train'])
 dl=base.loader(torch,ds,plan,{'num_workers':0,'pin_memory':False,'persistent_workers':False,'prefetch_factor':None})
 cfg={'lambda':LAM,'human_approved':True,'steps':200,'seed':1149,'initial_model_sha256':init,
 'initialization':'fresh official weights_init; hash matches baseline random initialization; no checkpoint load',
 'optimizer':'Adam','lr':.001,'weight_decay':.0001,'clip':.1,'batch':1,'crop':256,'exposure':.5,
 'RD_VS':[1,1],'upper':'mean(relu(original_preclamp_z-1)^2)','plan':plan,'split_counts':{k:len(v) for k,v in split.items()},
 'gradient_monitor':'upper parameter gradients every step; original gradient comparison steps1,10,20,...200; all preclip; autograd.grad does not accumulate',
 'script_sha256':base.sha(Path(__file__)),'adapter_sha256':check['adapter_sha256'],'loss_expressions':expressions,
 'no_dev':True,'no_checkpoint_saved':True,'protected_hashes':check['protected_hashes']}
 dump(EXP/'config/smoke_run.json',cfg)
 rows=[];start=time.perf_counter();iterator=iter(dl)
 with (EXP/'metrics/smoke_steps.jsonl').open('x',encoding='utf8') as log:
  for step in range(1,201):
   t=time.perf_counter();X,G,paths,xx,yy=next(iterator);wait=(time.perf_counter()-t)*1000;p=paths[0]
   assert p==split['train'][plan[step-1][0]] and p not in split['dev'] and p not in split['quarantine']
   torch.cuda.synchronize();t=time.perf_counter();x=X.cuda();g=G.cuda();torch.cuda.synchronize();transfer=(time.perf_counter()-t)*1000
   assert bool(x.any()) and bool(torch.isfinite(x).all()) and bool(torch.isfinite(g).all())
   before=[v.detach().clone() for v in params];torch.cuda.reset_peak_memory_stats()
   torch.cuda.synchronize();t=time.perf_counter()
   (y,c,b,k),z=forward_with_original_z(model,x)
   assert torch.equal(y,z.clamp(0,1))
   rd,vs,original=fn(x,g,y,b,k,.5);upper=upper_loss(z);weighted=LAM*upper;total=original+weighted
   torch.cuda.synchronize();forward=(time.perf_counter()-t)*1000
   assert all(bool(torch.isfinite(v).all()) for v in [y,z,rd,vs,upper,weighted,total])
   t=time.perf_counter()
   gu=torch.autograd.grad(upper,params,retain_graph=True)
   assert all(bool(torch.isfinite(v).all()) for v in gu)
   un=norm(gu);ratio=None;bn=None;cosine=None
   if step==1 or step%10==0:
    gb=torch.autograd.grad(original,params,retain_graph=True);bn=norm(gb)
    assert all(bool(torch.isfinite(v).all()) for v in gb)
    ratio=LAM*un/bn if bn else None
    cosine=sum(float((a.double()*d.double()).sum()) for a,d in zip(gu,gb))/(un*bn) if un and bn else None
    del gb
   torch.cuda.synchronize();monitor=(time.perf_counter()-t)*1000
   optimizer.zero_grad();t=time.perf_counter();total.backward();torch.cuda.synchronize();backward=(time.perf_counter()-t)*1000
   assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in params)
   t=time.perf_counter();gn=torch.nn.utils.clip_grad_norm_(params,.1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();update=(time.perf_counter()-t)*1000
   delta=sum(float((v.detach()-old).abs().sum()) for v,old in zip(params,before))
   assert math.isfinite(delta) and delta>0 and all(bool(torch.isfinite(v).all()) for v in params)
   row={'step':step,'source':p,'crop':ds.records[-1],'total':float(total.detach()),'RD':float(rd.detach()),'VS':float(vs.detach()),
   'upper':float(upper.detach()),'weighted_upper':float(weighted.detach()),'upper_grad_l2':un,'upper_grad_finite':True,
   'original_grad_l2_sampled':bn,'weighted_upper_to_original_grad_rms_ratio':ratio,'gradient_cosine_sampled':cosine,
   'total_grad_norm_before_clip':float(gn),'total_gradient_finite':True,'parameter_update_l1':delta,
   'pre_gt1_channel_ratio':float((z>1).float().mean()),'pre_lt0_channel_ratio':float((z<0).float().mean()),
   'data_ms':wait,'transfer_ms':transfer,'forward_loss_ms':forward,'gradient_monitor_ms':monitor,'backward_ms':backward,'clip_optimizer_ms':update,
   'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
   rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
   del X,G,x,g,y,c,b,k,z,rd,vs,original,upper,weighted,total,gu,before
   if step%25==0:print('UPDATE',step,'mean25total',statistics.mean(r['total'] for r in rows[-25:]),flush=True)
 elapsed=time.perf_counter()-start
 try:next(iterator);raise RuntimeError('Unexpected extra batch')
 except StopIteration:pass
 for p,h in cfg['protected_hashes'].items():assert base.sha(ROOT/p)==h,p
 assert len(rows)==200 and any(r['upper']>0 and r['upper_grad_l2']>0 for r in rows)
 ratio_rows=[r for r in rows if r['original_grad_l2_sampled'] is not None]
 ratios=[r['weighted_upper_to_original_grad_rms_ratio'] for r in ratio_rows if r['weighted_upper_to_original_grad_rms_ratio'] is not None]
 windows=[{'steps':[a+1,a+50],**{k:statistics.mean(r[k] for r in rows[a:a+50]) for k in ['total','RD','VS','upper','weighted_upper','pre_gt1_channel_ratio','pre_lt0_channel_ratio']}} for a in range(0,200,50)]
 summary={'status':'PASS','effective_updates':200,'unique_train_images':len({r['source'] for r in rows}),'lambda':LAM,
 'loss_stats':{k:stats([r[k] for r in rows]) for k in ['total','RD','VS','upper','weighted_upper']},
 'upper_active_steps':sum(r['upper']>0 for r in rows),'upper_active_fraction':sum(r['upper']>0 for r in rows)/200,
 'gradient_ratio_stats':stats(ratios),'gradient_samples':len(ratio_rows),
 'pooled_gradient_rms_ratio':LAM*math.sqrt(sum(r['upper_grad_l2']**2 for r in ratio_rows)/sum(r['original_grad_l2_sampled']**2 for r in ratio_rows)),
 'rolling50_windows':windows,'crop_resampling':base.crop_stats(ds.records),
 'elapsed_seconds':elapsed,'data_seconds':sum(r['data_ms'] for r in rows)/1000,'transfer_seconds':sum(r['transfer_ms'] for r in rows)/1000,
 'compute_seconds':sum(r['forward_loss_ms']+r['backward_ms']+r['clip_optimizer_ms'] for r in rows)/1000,
 'gradient_monitor_seconds':sum(r['gradient_monitor_ms'] for r in rows)/1000,
 'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rows),'peak_reserved_mib':max(r['peak_reserved_mib'] for r in rows),
 'nan_inf':False,'oom':False,'all_updates_finite_nonzero':True,'protected_hashes_unchanged':True,
 'clamp_equality_each_step':True,'dev_used':False,'checkpoint_saved':False,'no_quality_or_convergence_claim':True}
 dump(EXP/'metrics/smoke_summary.json',summary);dump(EXP/'metrics/smoke_crop_records.json',ds.records)
 text=['# Approved 200-step smoke test','',
 'PASS: 200 effective updates, fresh initialization seed1149; no checkpoint loaded or saved. No dev evaluation or lambda selection. Lambda fixed to '+repr(LAM)+'. No continuation.',
 '', 'Original network/RD/VS unchanged; observed original clamp input z used by the only additional squared upper penalty. Every step clamp equality verified; lambda0 forward/backward equality was established in the preceding gradient audit. Positive lambda intentionally changes the total gradient and trajectory.',
 '', 'Upper gradients measured every step; baseline comparison at1,10,...200 before clipping. For the same parameter count, the L2 norm ratio equals per-parameter-element RMS ratio. Per-step ratios and pooled-across-samples RMS ratio are distinct. autograd.grad monitoring does not accumulate into .grad. Total backward occurs once per update.',
 '', 'Timing separates data wait, transfer, forward/backward/optimizer, extra gradient-monitor overhead; elapsed also contains finite checks and logging. Memory includes monitoring tensors.',
 '', 'All200 different images come from the770-entry train manifest; no dev/quarantine/official val/test. Crop protocol is reused unchanged. A zero resample count would mean no rejection occurred, not new proof of the retry branch.',
 '', 'See config/smoke_run.json, metrics/smoke_steps.jsonl, metrics/smoke_summary.json and metrics/smoke_crop_records.json.',
 '', '## 50-step descriptive windows','',
 '|Steps|Total|RD|VS|Upper|Weighted upper|z>1 channels|','|---|---:|---:|---:|---:|---:|---:|']
 for w in windows:text.append(f"|{w['steps']}|{w['total']:.8f}|{w['RD']:.8f}|{w['VS']:.8f}|{w['upper']:.8f}|{w['weighted_upper']:.8f}|{w['pre_gt1_channel_ratio']:.6%}|")
 text+=['','Different windows contain different training images/crops; changes do not establish a fixed-input improvement or superiority over baseline. Loss fluctuations do not establish convergence. Gradient ratios may drift substantially from the initialization calibration.',
 '', 'Engineering smoke passed only. No formal15-epoch run, best selection, commit or push. Await human review.']
 (EXP/'SMOKE_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf8')
 with (EXP/'README.md').open('a',encoding='utf8') as f:f.write('\n## 后续人工批准与200步结果\n\n此前待确认状态为历史记录。用户现已批准lambda='+repr(LAM)+'；重新随机初始化的200步smoke已PASS，详见[SMOKE_RESULTS.md](SMOKE_RESULTS.md)及metrics/smoke_summary.json。没有启动正式训练或commit/push。\n')
 print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  p=EXP/'metrics/smoke_failure.json'
  if not p.exists():dump(p,{'status':'FAIL_STOPPED_NO_RETRY','error':repr(e),'traceback':traceback.format_exc()})
  raise
