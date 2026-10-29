"""EXP005A: fixed 15-epoch adapted baseline; preregistered dev-only gain diagnostics."""
import argparse,ast,collections,ctypes,gc,hashlib,importlib,importlib.metadata,json,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/exp005a_baseline_30ep';OUT=ROOT/'results/demos/di_retinex_exp005a';CK=ROOT/'checkpoints/di_retinex/exp005a';REPO=ROOT/'_model_audit/DI-retinex-low-light-enhancement';COMMIT='3f94f95ffb181904a837532a70adc064bee793a2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def memory():
 class M(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ['total','available','tp','ap','tv','av','ext']]
 m=M();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m));return {'total_gib':m.total/2**30,'available_gib':m.available/2**30}
def setup():
 import torch
 assert Path(sys.prefix).name.lower()=='nightuav' and torch.cuda.is_available()
 assert str(torch.__version__)=='2.13.0+cu130' and importlib.metadata.version('torchvision')=='0.28.0+cu130'
 g=['git','-c','safe.directory='+REPO.as_posix(),'-C',str(REPO)]
 assert subprocess.check_output(g+['rev-parse','HEAD'],text=True).strip()==COMMIT
 for n in ['mymodel.py','dataloader.py','lowlight_train.py']:
  assert subprocess.check_output(g+['hash-object',str(REPO/n)],text=True).strip()==subprocess.check_output(g+['rev-parse','HEAD:'+n],text=True).strip()
 split={n:(ROOT/f'configs/splits/u3d_{n}_v2_1.txt').read_text(encoding='utf-8').splitlines() for n in ['train','dev','quarantine']}
 for n,size in [('train',770),('dev',8),('quarantine',22)]:assert len(split[n])==len(set(split[n]))==size
 assert all(not set(split[a])&set(split[b]) for a,b in [('train','dev'),('train','quarantine'),('dev','quarantine')])
 meta=json.loads((ROOT/'configs/splits/u3d_train_dev_v2_1.json').read_text(encoding='utf-8'))
 for n in split:
  for p in split[n]:assert (ROOT/p).is_file() and (ROOT/p).resolve().is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve()) and sha(ROOT/p)==meta['image_sha256'][p]
 sys.path.insert(0,str(REPO));official=importlib.import_module('lowlight_train');return torch,official,split,meta
class PatchDataset:
 def __init__(self,paths):self.paths=paths;self.original=None
 def __len__(self):return len(self.paths)
 def __getitem__(self,spec):
  # Crop coordinates are created in the main process, independent of worker RNG/order.
  idx,x,y=spec
  if self.original is None:
   sys.path.insert(0,str(REPO));mod=importlib.import_module('dataloader');self.original=mod.lowlight_loader.__new__(mod.lowlight_loader);self.original.data_list=[str(ROOT/p) for p in self.paths]
  X,G,path=self.original[idx];assert tuple(X.shape)==(3,2160,3840)
  return X[:,y:y+256,x:x+256].contiguous(),G[:,y:y+256,x:x+256].contiguous(),self.paths[idx],x,y
 def __getstate__(self):return {'paths':self.paths,'original':None}
def loader(torch,ds,plan,cfg):
 kw=dict(batch_size=1,sampler=plan,num_workers=cfg['num_workers'],pin_memory=cfg['pin_memory'],generator=torch.Generator().manual_seed(1149))
 if cfg['num_workers']:kw.update(persistent_workers=cfg['persistent_workers'],prefetch_factor=cfg['prefetch_factor'])
 return torch.utils.data.DataLoader(ds,**kw)
def shutdown(dl):
 it=getattr(dl,'_iterator',None)
 if it is not None:it._shutdown_workers()
def make_loss(torch):
 from torch.distributions.normal import Normal
 tree=ast.parse((REPO/'lowlight_train.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');keys=['mask','Gaussian','target_exposure','ratio','loss1','loss2','loss'];a={n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys};assert set(a)==set(keys)
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['img_lowlight','img_gauss','enhanced_image','b','k','exp']],kwonlyargs=[],kw_defaults=[],defaults=[]);body=[a[k] for k in keys]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['loss1','loss2','loss']],ctx=ast.Load()))];f=ast.FunctionDef(name='loss_fn',args=args,body=body,decorator_list=[]);scope={'torch':torch,'Normal':Normal};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(REPO/'lowlight_train.py'),'exec'),scope);return scope['loss_fn'],{k:ast.unparse(a[k]) for k in keys}
def train():
 import numpy as np,cv2
 from PIL import Image,ImageDraw
 torch,official,split,meta=setup();benchmark_source=ROOT/'experiments/exp005a0_baseline_pilot/metrics/loader_benchmark.json';bench=json.loads(benchmark_source.read_text());cfg=bench['selected'];assert cfg['num_workers']==0;assert (EXP/'budget_stop_guard.json').exists() and (CK/'epoch010.pth').exists() and not (EXP/'config/continuation_to15.json').exists()
 assert subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'check-ignore','-q','results/demos/di_retinex_exp005a/probe.png']).returncode==0
 official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);loss_fn,expressions=make_loss(torch)
 source_hashes={n:sha(REPO/n) for n in ['mymodel.py','dataloader.py','lowlight_train.py']};split_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'configs/splits').glob('*v2_1*')}
 config={'experiment':'EXP005A adapted baseline','epochs':15,'steps_per_epoch':770,'total_steps':11550,'seed':1149,'initialization':'random official weights_init; no prior checkpoint loaded','optimizer':'Adam','lr':.001,'weight_decay':.0001,'grad_clip':.1,'exposure_mean':.5,'exposure_std':.001,'RD_VS_weights':[1,1],'batch_size':1,'train_crop':[256,256],'crop_semantics':'full image original official loader including Gaussian then paired uniform crop; main-process coordinates independent of workers','sampling':'each epoch shuffled permutation of all 770; seed1149 separate sampler and crop RNG streams','loader':cfg,'dev_size':[1280,720],'dev_resize':'Pillow RGB LANCZOS','dev_epochs':list(range(16)),'dev_objective':'original RD+VS on resized RGB and GaussianBlur of resized uint8 RGB; fixed per-image stochastic exposure using isolated seed1149+i each epoch; unsupervised dev objective, not image quality score','saved_epochs':[0,1,2,5,10,15],'checkpoint_epochs':[1,2,5,10,15],'best_selection':False,'region_diagnostics':{'luminance':'encoded RGB Yprime=.2126R+.7152G+.0722B','output':'official clamped enhanced RGB','dark_mask':'fixed resized input Y_in < 0.2','bright_mask':'fixed resized input Y_in > 0.7','gain':'Y_out - Y_in','statistics':['pixel_count','mean','median','p10','p90'],'empty_region':'count 0, all gain statistics null','purpose':'preregistered diagnostics only; not a loss, model component or quality score'},'model_commit':COMMIT,'source_sha256':source_hashes,'split_sha256':split_hashes,'script_sha256':sha(Path(__file__)),'loader_benchmark_source':benchmark_source.relative_to(ROOT).as_posix(),'loader_benchmark_sha256':sha(benchmark_source),'split_counts':{k:len(v) for k,v in split.items()},'torch':str(torch.__version__),'torchvision':importlib.metadata.version('torchvision'),'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'loss_expressions':expressions,'rolling_window':50,'command':[sys.executable,'-B',str(Path(__file__))],'official_protocol_difference':'Official loader uses whole images; 3840x2160 U3D is adapted to 256x256 patches. This is not strict paper reproduction.'}
 # Same-run continuation only. Reconstruct initialization and original CPU Normal draws.
 original_config=json.loads((EXP/'config/run.json').read_text(encoding='utf-8'))
 assert sha(EXP/'config/original_30ep_runner.py')==original_config['script_sha256']
 original_rows=[json.loads(line) for line in (EXP/'metrics/train_steps.jsonl').read_text().splitlines()]
 completed_rows=[x for x in original_rows if x['global_step']<=7700]
 replay_reference={x['global_step']:x for x in original_rows if x['global_step']>7700}
 assert len(completed_rows)==7700 and replay_reference and max(replay_reference)<8471
 for epoch in range(1,11):assert {x['source'] for x in completed_rows if x['epoch']==epoch}==set(split['train'])
 initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
 assert initial_hash==original_config['initial_model_sha256'],'Initialization reconstruction mismatch'
 from torch.distributions.normal import Normal
 distribution=Normal(torch.tensor([.5]),torch.tensor([.001]))
 for _ in range(7700):distribution.sample()
 resume_state=torch.load(CK/'epoch010.pth',map_location='cpu',weights_only=True)
 assert resume_state['epoch']==10 and resume_state['source_commit']==COMMIT
 model.load_state_dict(resume_state['model'],strict=True);optimizer.load_state_dict(resume_state['optimizer'])
 assert all(torch.equal(v.detach().cpu(),resume_state['model'][k]) for k,v in model.state_dict().items())
 assert optimizer.param_groups[0]['lr']==.001 and optimizer.param_groups[0]['weight_decay']==.0001
 config['continuation']={'same_experiment_checkpoint':'checkpoints/di_retinex/exp005a/epoch010.pth','checkpoint_sha256':sha(CK/'epoch010.pth'),'pilot_weights_used':False,'resume_epoch':11,'stop_epoch':15,'reason':'User lowered budget during execution','original_run_config_sha256':sha(EXP/'config/run.json'),'initial_random_model_reconstruction_matches':True,'cpu_rng_reconstruction':'same initialization then original Normal(.5,.001) sample advanced 7700 times; CUDA RNG unchanged by original training','partial_epoch11_steps_to_replay':len(replay_reference),'verification':'Require exact loss/update and sample/crop agreement for all partial steps before continuing'}
 config['initialization']='Overall run began randomly at E0. This continuation restores only its own E10 model and Adam state.'
 config['started_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();config['initial_model_sha256']=initial_hash;config['resumed_model_sha256']=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest();dump(EXP/'config/continuation_to15.json',config)
 dev_inputs=[]
 for p in split['dev']:
  with Image.open(ROOT/p) as im:assert im.size==(3840,2160);a=np.array(im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS))
  g=cv2.GaussianBlur(a,(9,9),sigmaX=1.7,sigmaY=1.7);X=torch.from_numpy(a/255.).float().permute(2,0,1).unsqueeze(0).contiguous();G=torch.from_numpy(g/255.).float().permute(2,0,1).unsqueeze(0).contiguous();dev_inputs.append((p,X,G,a))
 dev_input_records=[];dev_region_inputs=[]
 for p,X,G,a in dev_inputs:
  yin=(X[0]*torch.tensor([.2126,.7152,.0722])[:,None,None]).sum(0)
  dev_region_inputs.append((yin,yin<.2,yin>.7))
  dev_input_records.append({'source':p,'resized_rgb_sha256':hashlib.sha256(a.tobytes()).hexdigest(),'dark_pixel_count':int((yin<.2).sum()),'bright_pixel_count':int((yin>.7).sum()),'mean_input_luminance':float(yin.mean())})
 assert dev_input_records==json.loads((EXP/'config/fixed_dev_inputs.json').read_text())
 def region_gain(yin,yout,mask):
  values=(yout-yin)[mask];n=values.numel()
  if not n:return {'pixel_count':0,'mean':None,'median':None,'p10':None,'p90':None}
  q=torch.quantile(values,torch.tensor([.1,.5,.9],device=values.device))
  return {'pixel_count':n,'mean':float(values.mean()),'median':float(q[1]),'p10':float(q[0]),'p90':float(q[2])}
 dev_history=[json.loads((EXP/f'metrics/dev_epoch{i:03d}.json').read_text()) for i in range(11)];train_history=[json.loads((EXP/f'metrics/train_epoch{i:03d}.json').read_text())['summary'] for i in range(1,11)];checkpoint_checks=[];rolling=collections.deque(([x['loss_total'],x['loss_rd'],x['loss_vs']] for x in completed_rows[-50:]),maxlen=50);saved_epochs={0,1,2,5,10,15}
 def evaluate(epoch):
  before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone();model.eval();rows=[];start=time.perf_counter()
  with torch.no_grad():
   for i,(p,X,G,a) in enumerate(dev_inputs):
    x=X.cuda();g=G.cuda();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();y,c,b,k=model(x);torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000;z=k*(x+.5*(b-1))+.5*(b+1);assert torch.equal(y,z.clamp(0,1))
    with torch.random.fork_rng(devices=[0]):
     torch.manual_seed(1149+i);torch.cuda.manual_seed(1149+i);rd,vs,obj=loss_fn(x,g,y,b,k,.5)
    assert all(torch.isfinite(v).all().item() for v in [y,c,b,k,z,rd,vs,obj]),'Nonfinite dev'
    luma=(y[0]*torch.tensor([.2126,.7152,.0722],device='cuda')[:,None,None]).sum(0);quant=torch.quantile(luma.flatten(),torch.tensor([.5,.95],device='cuda'))
    row={'epoch':epoch,'source':p,'raw_min':float(z.min()),'raw_max':float(z.max()),'raw_lt0_ratio':float((z<0).float().mean()),'raw_gt1_ratio':float((z>1).float().mean()),'clamped_zero_ratio':float((y==0).float().mean()),'clamped_one_ratio':float((y==1).float().mean()),'mean_luminance':float(luma.mean()),'median_luminance':float(quant[0]),'p95_luminance':float(quant[1]),'unsupervised_dev_objective':float(obj),'unsupervised_dev_RD':float(rd),'unsupervised_dev_VS':float(vs),'finite':True,'forward_ms':ms,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    yin=dev_region_inputs[i][0].cuda()
    row['dark_region_gain']=region_gain(yin,luma,dev_region_inputs[i][1].cuda());row['bright_region_gain']=region_gain(yin,luma,dev_region_inputs[i][2].cuda())
    row['mean_rgb']=[float(v) for v in y.mean((0,2,3))]
    assert row['dark_region_gain']['pixel_count']==dev_input_records[i]['dark_pixel_count'] and row['bright_region_gain']['pixel_count']==dev_input_records[i]['bright_pixel_count']
    del yin
    if epoch in saved_epochs:
     target=OUT/f'epoch{epoch:03d}'/(Path(p).stem+'.png');target.parent.mkdir(exist_ok=True)
     with target.open('xb') as f:Image.fromarray(np.rint(y[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(f,format='PNG')
     with Image.open(target) as im:assert im.size==(1280,720) and im.mode=='RGB';im.verify()
    rows.append(row);del x,g,y,c,b,k,z,luma,quant,rd,vs,obj
  assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items());assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state());model.train()
  result={'epoch':epoch,'images':rows,'seconds':time.perf_counter()-start,'no_parameter_or_rng_update':True};dev_history.append(result);dump(EXP/f'metrics/dev_epoch{epoch:03d}.json',result);print('DEV',epoch,'PASS',flush=True)
 def save_checkpoint(epoch):
  path=CK/f'epoch{epoch:03d}.pth';state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  with path.open('xb') as f:torch.save({'purpose':'adapted_baseline_no_best_selection','epoch':epoch,'model':state,'optimizer':optimizer.state_dict(),'seed':1149,'source_commit':COMMIT},f)
  loaded=torch.load(path,map_location='cpu',weights_only=True)
  with torch.random.fork_rng(devices=[0]):
   fresh=official.mymodel.enhance_net_nopool();fresh.load_state_dict(loaded['model'],strict=True);assert all(torch.equal(v,state[k]) for k,v in fresh.state_dict().items());fresh=fresh.cuda().eval();model.eval()
   with torch.no_grad():
    x=dev_inputs[0][1].cuda();a=fresh(x)[0];b=model(x)[0];torch.cuda.synchronize();assert torch.isfinite(a).all() and torch.equal(a,b)
   del fresh,x,a,b
  model.train();checkpoint_checks.append({'epoch':epoch,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'weights_only':True,'strict':True,'state_exact':True,'forward_exact':True});print('CHECKPOINT',epoch,'PASS',flush=True)
 # Revalidate existing checkpoints without altering resumed RNG.
 for old_epoch in [1,2,5,10]:
  path=CK/f'epoch{old_epoch:03d}.pth';state=torch.load(path,map_location='cpu',weights_only=True)
  with torch.random.fork_rng(devices=[0]):
   verify_model=official.mymodel.enhance_net_nopool();verify_model.load_state_dict(state['model'],strict=True)
   assert all(torch.equal(v,state['model'][k]) for k,v in verify_model.state_dict().items())
   verify_model=verify_model.cuda().eval()
   with torch.no_grad():
    test_y=verify_model(dev_inputs[0][1].cuda())[0];assert torch.isfinite(test_y).all()
   del verify_model,test_y
  checkpoint_checks.append({'epoch':old_epoch,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'weights_only':True,'strict':True,'state_exact':True,'forward_finite':True,'verification':'original run plus independent safe reload during limit change'})
 total_start=time.perf_counter();process_start=time.process_time()
 plan=[];dataset=PatchDataset(split['train']);dl=loader(torch,dataset,plan,cfg);sampler=random.Random(1149);crop=random.Random(1149)
 # Advance independent sampler/crop/loader RNGs to the E11 boundary.
 for prior_epoch in range(10):
  previous=list(range(770));sampler.shuffle(previous)
  for _ in previous:crop.randint(0,3840-256);crop.randint(0,2160-256)
  iter(dl)
 with (EXP/'metrics/train_steps_15ep.jsonl').open('x',encoding='utf-8') as log:
  for record in completed_rows:log.write(json.dumps(record,allow_nan=False)+'\n')
  log.flush()
  for epoch in range(11,16):
   order=list(range(770));sampler.shuffle(order);plan[:]=[(idx,crop.randint(0,3840-256),crop.randint(0,2160-256)) for idx in order];start=time.perf_counter();t=time.perf_counter();iterator=iter(dl);iterator_ms=(time.perf_counter()-t)*1000;records=[]
   for localstep in range(1,771):
    t=time.perf_counter();X,G,paths,xx,yy=next(iterator);wait_ms=(time.perf_counter()-t)*1000;source=paths[0];assert source==split['train'][order[localstep-1]] and source not in split['dev'] and source not in split['quarantine']
    torch.cuda.synchronize();t=time.perf_counter();X=X.cuda(non_blocking=cfg['pin_memory']);G=G.cuda(non_blocking=cfg['pin_memory']);torch.cuda.synchronize();transfer_ms=(time.perf_counter()-t)*1000
    assert X.device.type=='cuda' and X.shape==(1,3,256,256) and torch.isfinite(X).all() and torch.isfinite(G).all() and X.mean()>0
    before=[p.detach().clone() for p in model.parameters()];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();y,c,b,k=model(X);rd,vs,loss=loss_fn(X,G,y,b,k,.5);torch.cuda.synchronize();fwd=(time.perf_counter()-t)*1000;assert all(torch.isfinite(v).all().item() for v in [y,c,b,k,rd,vs,loss])
    optimizer.zero_grad();t=time.perf_counter();loss.backward();torch.cuda.synchronize();bwd=(time.perf_counter()-t)*1000;assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    t=time.perf_counter();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();update=(time.perf_counter()-t)*1000
    delta=sum(float((p.detach()-v).abs().sum()) for p,v in zip(model.parameters(),before));assert delta>0 and all(torch.isfinite(p).all().item() for p in model.parameters())
    losses=[float(loss.detach()),float(rd.detach()),float(vs.detach())];rolling.append(losses)
    row={'epoch':epoch,'step':localstep,'global_step':(epoch-1)*770+localstep,'source':source,'crop_xywh':[int(xx[0]),int(yy[0]),256,256],'loss_total':losses[0],'loss_rd':losses[1],'loss_vs':losses[2],'rolling50_total':statistics.mean(v[0] for v in rolling),'rolling50_rd':statistics.mean(v[1] for v in rolling),'rolling50_vs':statistics.mean(v[2] for v in rolling),'gradient_finite':True,'gradient_norm_before_clip':float(grad),'parameter_update_l1':delta,'cuda':True,'wait_ms':wait_ms,'transfer_ms':transfer_ms,'forward_loss_ms':fwd,'backward_ms':bwd,'clip_optimizer_ms':update,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
    if row['global_step'] in replay_reference:
     reference=replay_reference[row['global_step']]
     for key in ['source','crop_xywh','loss_total','loss_rd','loss_vs','gradient_norm_before_clip','parameter_update_l1','rolling50_total','rolling50_rd','rolling50_vs']:assert row[key]==reference[key],f'Continuation replay mismatch at {row["global_step"]}: {key}'
     if row['global_step']==max(replay_reference):
      dump(EXP/'config/continuation_replay_verified.json',{'verified':True,'steps':len(replay_reference),'global_steps':sorted(replay_reference),'exact_fields':['source','crop','loss RD/VS/total','gradient norm','parameter update L1','rolling50'],'stop_epoch':15});print('EXACT CONTINUATION REPLAY VERIFIED',flush=True)
    records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();del X,G,y,c,b,k,rd,vs,loss,before
    if localstep%50==0:print('TRAIN',epoch,localstep,'rolling50',row['rolling50_total'],flush=True)
   try:next(iterator);raise AssertionError('Extra batch')
   except StopIteration:pass
   assert len(records)==770 and {r['source'] for r in records}==set(split['train'])
   end=time.perf_counter();summary={'epoch':epoch,'steps':770,'unique_images':770,'all_train_covered_once':True,'mean_total':statistics.mean(x['loss_total'] for x in records),'mean_RD':statistics.mean(x['loss_rd'] for x in records),'mean_VS':statistics.mean(x['loss_vs'] for x in records),'train_seconds':end-start,'iterator_setup_ms':iterator_ms,'mean_wait_ms':statistics.mean(x['wait_ms'] for x in records),'compute_seconds':sum(x['forward_loss_ms']+x['backward_ms']+x['clip_optimizer_ms'] for x in records)/1000,'transfer_seconds':sum(x['transfer_ms'] for x in records)/1000,'peak_allocated_mib':max(x['peak_allocated_mib'] for x in records),'peak_reserved_mib':max(x['peak_reserved_mib'] for x in records),'all_gradients_finite':True,'all_parameters_updated':True,'nan_inf':False}
   train_history.append(summary);dump(EXP/f'metrics/train_epoch{epoch:03d}.json',{'summary':summary,'steps':records});print('EPOCH',json.dumps(summary),flush=True);evaluate(epoch)
   if epoch in [1,2,5,10,15]:save_checkpoint(epoch)
 shutdown(dl);total_seconds=time.perf_counter()-total_start
 dump(EXP/'metrics/train_history.json',{'epochs':train_history,'step_log':'train_steps_15ep.jsonl','rolling_definition':'last up to 50 global steps, crosses epoch boundaries'});dump(EXP/'metrics/dev_history.json',dev_history)
 # Fixed unranked montage: input followed by five saved epochs, identical RGB thumbnails.
 montages=[]
 for p,X,G,a in dev_inputs:
  canvas=Image.new('RGB',(7*480,300),'#20242a');draw=ImageDraw.Draw(canvas);images=[Image.fromarray(a)]
  for epoch in [0,1,2,5,10,15]:
   with Image.open(OUT/f'epoch{epoch:03d}'/(Path(p).stem+'.png')) as im:images.append(im.copy())
  for j,(im,label) in enumerate(zip(images,['input','epoch0','epoch1','epoch2','epoch5','epoch10','epoch15'])):
   im=im.resize((480,270),Image.Resampling.LANCZOS);canvas.paste(im,(j*480,30));draw.text((j*480+8,8),Path(p).name+' | '+label,fill='white')
  target=OUT/'montage'/(Path(p).stem+'.png');target.parent.mkdir(exist_ok=True);canvas.save(target)
  with Image.open(target) as im:assert im.size==(3360,300);im.verify()
  montages.append(target.relative_to(ROOT).as_posix())
 assert len(train_history)==15 and sum(x['steps'] for x in train_history)==11550 and len(dev_history)==16 and [x['epoch'] for x in checkpoint_checks]==[1,2,5,10,15]
 assert all(sha(REPO/n)==h for n,h in source_hashes.items()) and all(sha(ROOT/p)==h for p,h in split_hashes.items())
 assert all(sha(ROOT/p)==h for p,h in meta['image_sha256'].items())
 final={'status':'COMPLETED_15_EPOCH_ADAPTED_BASELINE','epochs':15,'steps':11550,'training_seconds':sum(x['train_seconds'] for x in train_history),'continuation_training_evaluation_checkpoint_seconds':total_seconds,'original_recorded_training_seconds':sum(x['train_seconds'] for x in train_history[:10]),'continuation_process_cpu_seconds':time.process_time()-process_start,'finished_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'timing_warning':'wall intervals may include sleep/stalls; process CPU time is not GPU active time','total_scope':'continuation wall time is E11..E15 only; per-epoch train sum covers E1..E15; old partial E11 work is preserved separately and excluded from logical epoch totals','checkpoint_checks':checkpoint_checks,'montages':montages,'nan_inf':False,'oom':False,'all_770_covered_each_epoch':True,'no_train_dev_quarantine_mixing':True,'source_and_input_hashes_unchanged':True,'best_checkpoint_selected':False}
 dump(EXP/'metrics/summary.json',final)
 text='# EXP005A: DI-Retinex-U3D 15-epoch adapted baseline\n\nCompleted exactly 15 epochs, 770 images per epoch, 11550 updates. STOPPED for human research review; no extension, early stopping or best checkpoint selection.\n\nOfficial DI-Retinex dataloader uses full images. U3D originals are 3840x2160; this project uses 256x256 paired random patches after the original full-image Gaussian processing and conversion. Therefore this is an adapted baseline, not strict paper reproduction and not an innovation experiment.\n\nOriginal network and RD+VS 1:1, random official initialization, seed1149, Adam lr0.001, weight decay0.0001, clipping0.1, exposure0.5 (original Normal std0.001), batch1. Overall training began with fresh random initialization, without pilot/pretrained weights. The user-requested budget reduction restores this SAME experiment E10 checkpoint and reconstructs RNG/sampler states, verified against archived partial E11 steps. All 770 train paths traversed once every epoch; dev8 and quarantine22 excluded. Loader settings and source/split/script hashes: config/run.json. Loader configuration is inherited from the pilot benchmark referenced by hash in config/run.json; no prior weights are used. Only tested configurations qualify for selection; skipped RAM-heavy configurations are not claimed slower.\n\nDev0..15: full eight RGB images resized with LANCZOS to 1280x720, no crop. Eval/no_grad, parameters and training RNG unchanged; train mode restored. Raw output reconstructed and checked against official clamp. Ratios count channels. Luminance is encoded RGB Yprime, not linear-light. Unsupervised dev objective uses original RD/VS on resized inputs and their full Gaussian blur, fixed per-image exposure sample via isolated RNG for epoch comparability. It is NOT an image quality score or GT metric.\n\nEpoch0/1/2/5/10/15 images and eight input|epoch0|epoch1|epoch2|epoch5|epoch10|epoch15 montages are Git ignored under results/demos/di_retinex_exp005a/. No automatic preference/ranking.\n\nCheckpoints epoch001/002/005/010/015 under checkpoints/di_retinex/exp005a/ are unranked adapted baseline checkpoints; weights_only and strict state loading plus exact forward match verified. No best.pth.\n\nMetrics include per-epoch average total/RD/VS, global rolling50 step losses, per-step image/crop, finite gradients and parameter updates. Synchronized GPU-phase wall timing excludes separate finite/parameter-delta checks. Loader wait is main-process blocking time; with workers it is not total CPU work. Timing scopes and startup included in benchmark are explicit; do not infer whole-training speedup from loader-only throughput. Reserved CUDA memory includes cached dev allocations. No training objective adaptation or augmentation changes were made.\n\nThis small dev set and limited scene metadata cannot establish universal scene independence or final generalization. Fifteen epochs do not establish convergence. Human review is required before choosing any longer protocol. No commit/push.\n\n## Actual epoch summaries\n\n|Epoch|Mean total|Mean RD|Mean VS|Train seconds|Mean wait ms|\n|---|---:|---:|---:|---:|---:|\n'
 for x in train_history:text+=f"|{x['epoch']}|{x['mean_total']:.8f}|{x['mean_RD']:.8f}|{x['mean_VS']:.8f}|{x['train_seconds']:.3f}|{x['mean_wait_ms']:.3f}|\n"
 text+='\nEffective budget: 15 epochs. Directory name retained for provenance. Original config/run.json records the superseded 30-epoch launch; config/continuation_to15.json and config/continuation_replay_verified.json document the amendment. Original train_steps.jsonl is untouched; train_steps_15ep.jsonl is the canonical 11550-step history. No epochs 16..30 are run.\n\nDark/bright gain diagnostics use fixed resized input masks Y_in<0.2 and Y_in>0.7, with clamped output Y_out. Counts and mean/median/p10/p90 gains are recorded per image and epoch; empty masks yield null gains. These preregistered thresholds are diagnostics only, not model/loss changes. Fixed input RGB hashes and mask counts are in config/fixed_dev_inputs.json.\n\nFinal execution summary: metrics/summary.json. No best epoch conclusion.\n'
 with (EXP/'README.md').open('x',encoding='utf-8') as f:f.write(text)
 import report_exp005a_curves
 report_exp005a_curves.main()
 print('BASELINE COMPLETE',json.dumps(final),flush=True)
if __name__=='__main__':
 try:train()
 except Exception as e:
  if EXP.exists() and not (EXP/'failure.json').exists():dump(EXP/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'status':'STOPPED_FAILURE_NO_RETRY'})
  raise
