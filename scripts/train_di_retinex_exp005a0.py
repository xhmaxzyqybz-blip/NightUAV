"""EXP005A-0: fixed 10-epoch adapted baseline pilot; benchmark loader engineering only."""
import argparse,ast,collections,ctypes,gc,hashlib,importlib,importlib.metadata,json,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/exp005a0_baseline_pilot';OUT=ROOT/'results/demos/di_retinex_exp005a0';CK=ROOT/'checkpoints/di_retinex/exp005a0';REPO=ROOT/'_model_audit/DI-retinex-low-light-enhancement';COMMIT='3f94f95ffb181904a837532a70adc064bee793a2'
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
def benchmark():
 torch,official,split,meta=setup();assert not EXP.exists(),'Refuse overwrite benchmark';assert not OUT.exists() and not CK.exists()
 specs=[];rng=random.Random(1149);indices=list(range(770));rng.shuffle(indices)
 for idx in indices[:16]:specs.append((idx,rng.randint(0,3840-256),rng.randint(0,2160-256)))
 ds=PatchDataset(split['train']);references={}
 for idx,x,y in specs:
  a,b,p,_,_=ds[(idx,x,y)];references[p]=(a,b)
 configs=[dict(num_workers=0,pin_memory=False,persistent_workers=False,prefetch_factor=None),dict(num_workers=0,pin_memory=True,persistent_workers=False,prefetch_factor=None),dict(num_workers=1,pin_memory=True,persistent_workers=True,prefetch_factor=2),dict(num_workers=2,pin_memory=False,persistent_workers=False,prefetch_factor=2),dict(num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2),dict(num_workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=4)]
 results=[];torch.zeros(1,device='cuda');torch.cuda.synchronize()
 for cfg in configs:
  mem=memory();need=1.5+1.25*cfg['num_workers']
  if cfg['num_workers'] and mem['available_gib']<need:
   results.append({'config':cfg,'status':'skipped_memory_budget','memory':mem,'minimum_available_gib':need});print('BENCH SKIP',cfg,mem,flush=True);continue
  dl=loader(torch,ds,specs,cfg);passes=[]
  for repeat in range(2):
   start=time.perf_counter();it=iter(dl);validation=0.;transfer=0.;seen=[]
   for j in range(len(specs)):
    X,G,paths,xx,yy=next(it);p=paths[0];vt=time.perf_counter();assert torch.equal(X[0],references[p][0]) and torch.equal(G[0],references[p][1]);validation+=time.perf_counter()-vt
    t=time.perf_counter();X=X.cuda(non_blocking=cfg['pin_memory']);G=G.cuda(non_blocking=cfg['pin_memory']);torch.cuda.synchronize();transfer+=time.perf_counter()-t;seen.append(p);del X,G
   try:next(it);raise AssertionError('unexpected extra batch')
   except StopIteration:pass
   elapsed=time.perf_counter()-start;assert seen==[split['train'][i] for i,x,y in specs]
   passes.append({'repeat':repeat,'wall_seconds':elapsed,'equivalence_check_seconds':validation,'measured_seconds_excluding_comparison':elapsed-validation,'images_per_second':len(specs)/(elapsed-validation),'transfer_seconds':transfer,'bitwise_equivalent':True})
  shutdown(dl);del dl;gc.collect();results.append({'config':cfg,'status':'tested','memory':mem,'passes':passes});print('BENCH',cfg,passes,flush=True)
 tested=[x for x in results if x['status']=='tested'];winner=max(tested,key=lambda x:x['passes'][1]['images_per_second'])
 report={'selection_rule':'highest second-pass loader+transfer images/s; 16 train images, two passes; includes iterator start/finish, excludes bitwise comparison time; warm OS cache possible','scope':'loader-only microbenchmark; no model computation overlap measured','sample_plan':specs,'source_images':[split['train'][i] for i,x,y in specs],'candidates':results,'selected':winner['config'],'all_tested_candidates_bitwise_equivalent':True,'baseline_images_per_second':tested[0]['passes'][1]['images_per_second'],'selected_images_per_second':winner['passes'][1]['images_per_second'],'speedup_vs_benchmark_baseline':winner['passes'][1]['images_per_second']/tested[0]['passes'][1]['images_per_second'],'memory_policy':'Skip extra processes if free RAM <1.5GiB +1.25GiB per worker. Skipped configurations are not claimed slower.','script_sha256':sha(Path(__file__))}
 dump(EXP/'metrics/loader_benchmark.json',report);print('SELECTED',json.dumps(report['selected']),flush=True)
def make_loss(torch):
 from torch.distributions.normal import Normal
 tree=ast.parse((REPO/'lowlight_train.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train');keys=['mask','Gaussian','target_exposure','ratio','loss1','loss2','loss'];a={n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys};assert set(a)==set(keys)
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['img_lowlight','img_gauss','enhanced_image','b','k','exp']],kwonlyargs=[],kw_defaults=[],defaults=[]);body=[a[k] for k in keys]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['loss1','loss2','loss']],ctx=ast.Load()))];f=ast.FunctionDef(name='loss_fn',args=args,body=body,decorator_list=[]);scope={'torch':torch,'Normal':Normal};exec(compile(ast.fix_missing_locations(ast.Module(body=[f],type_ignores=[])),str(REPO/'lowlight_train.py'),'exec'),scope);return scope['loss_fn'],{k:ast.unparse(a[k]) for k in keys}
def train():
 import numpy as np,cv2
 from PIL import Image,ImageDraw
 torch,official,split,meta=setup();bench=json.loads((EXP/'metrics/loader_benchmark.json').read_text());cfg=bench['selected'];assert not (EXP/'config/run.json').exists() and not OUT.exists() and not CK.exists()
 assert subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'check-ignore','-q','results/demos/di_retinex_exp005a0/probe.png']).returncode==0
 official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);loss_fn,expressions=make_loss(torch)
 source_hashes={n:sha(REPO/n) for n in ['mymodel.py','dataloader.py','lowlight_train.py']};split_hashes={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in (ROOT/'configs/splits').glob('*v2_1*')}
 config={'experiment':'EXP005A-0 adapted baseline pilot','epochs':10,'steps_per_epoch':770,'total_steps':7700,'seed':1149,'initialization':'random official weights_init; no prior checkpoint loaded','optimizer':'Adam','lr':.001,'weight_decay':.0001,'grad_clip':.1,'exposure_mean':.5,'exposure_std':.001,'RD_VS_weights':[1,1],'batch_size':1,'train_crop':[256,256],'crop_semantics':'full image original official loader including Gaussian then paired uniform crop; main-process coordinates independent of workers','sampling':'each epoch shuffled permutation of all 770; seed1149 separate sampler and crop RNG streams','loader':cfg,'dev_size':[1280,720],'dev_resize':'Pillow RGB LANCZOS','dev_epochs':list(range(11)),'dev_objective':'original RD+VS on resized RGB and GaussianBlur of resized uint8 RGB; fixed per-image stochastic exposure using isolated seed1149+i each epoch; unsupervised dev objective, not image quality score','saved_epochs':[0,1,2,5,10],'checkpoint_epochs':[1,2,5,10],'best_selection':False,'model_commit':COMMIT,'source_sha256':source_hashes,'split_sha256':split_hashes,'script_sha256':sha(Path(__file__)),'loader_benchmark_sha256':sha(EXP/'metrics/loader_benchmark.json'),'split_counts':{k:len(v) for k,v in split.items()},'torch':str(torch.__version__),'torchvision':importlib.metadata.version('torchvision'),'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'loss_expressions':expressions,'rolling_window':50,'command':[sys.executable,'-B',str(Path(__file__)),'--train'],'official_protocol_difference':'Official loader uses whole images; 3840x2160 U3D is adapted to 256x256 patches. This is not strict paper reproduction.'}
 dump(EXP/'config/run.json',config);OUT.mkdir(parents=True);CK.mkdir(parents=True)
 dev_inputs=[]
 for p in split['dev']:
  with Image.open(ROOT/p) as im:assert im.size==(3840,2160);a=np.array(im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS))
  g=cv2.GaussianBlur(a,(9,9),sigmaX=1.7,sigmaY=1.7);X=torch.from_numpy(a/255.).float().permute(2,0,1).unsqueeze(0).contiguous();G=torch.from_numpy(g/255.).float().permute(2,0,1).unsqueeze(0).contiguous();dev_inputs.append((p,X,G,a))
 dev_history=[];train_history=[];checkpoint_checks=[];step_records=[];rolling=collections.deque(maxlen=50);saved_epochs={0,1,2,5,10}
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
    if epoch in saved_epochs:
     target=OUT/f'epoch{epoch:03d}'/(Path(p).stem+'.png');target.parent.mkdir(exist_ok=True)
     with target.open('xb') as f:Image.fromarray(np.rint(y[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')).save(f,format='PNG')
     with Image.open(target) as im:assert im.size==(1280,720) and im.mode=='RGB';im.verify()
    rows.append(row);del x,g,y,c,b,k,z,luma,quant,rd,vs,obj
  assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items());assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state());model.train()
  result={'epoch':epoch,'images':rows,'seconds':time.perf_counter()-start,'no_parameter_or_rng_update':True};dev_history.append(result);dump(EXP/f'metrics/dev_epoch{epoch:03d}.json',result);print('DEV',epoch,'PASS',flush=True)
 def save_checkpoint(epoch):
  path=CK/f'epoch{epoch:03d}.pth';state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  with path.open('xb') as f:torch.save({'purpose':'pilot_analysis_only_not_best','epoch':epoch,'model':state,'optimizer':optimizer.state_dict(),'seed':1149,'source_commit':COMMIT},f)
  loaded=torch.load(path,map_location='cpu',weights_only=True)
  with torch.random.fork_rng(devices=[0]):
   fresh=official.mymodel.enhance_net_nopool();fresh.load_state_dict(loaded['model'],strict=True);assert all(torch.equal(v,state[k]) for k,v in fresh.state_dict().items());fresh=fresh.cuda().eval();model.eval()
   with torch.no_grad():
    x=dev_inputs[0][1].cuda();a=fresh(x)[0];b=model(x)[0];torch.cuda.synchronize();assert torch.isfinite(a).all() and torch.equal(a,b)
   del fresh,x,a,b
  model.train();checkpoint_checks.append({'epoch':epoch,'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'weights_only':True,'strict':True,'state_exact':True,'forward_exact':True});print('CHECKPOINT',epoch,'PASS',flush=True)
 total_start=time.perf_counter();evaluate(0)
 plan=[];dataset=PatchDataset(split['train']);dl=loader(torch,dataset,plan,cfg);sampler=random.Random(1149);crop=random.Random(1149)
 with (EXP/'metrics/train_steps.jsonl').open('x',encoding='utf-8') as log:
  for epoch in range(1,11):
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
    records.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();del X,G,y,c,b,k,rd,vs,loss,before
    if localstep%50==0:print('TRAIN',epoch,localstep,'rolling50',row['rolling50_total'],flush=True)
   try:next(iterator);raise AssertionError('Extra batch')
   except StopIteration:pass
   assert len(records)==770 and {r['source'] for r in records}==set(split['train'])
   end=time.perf_counter();summary={'epoch':epoch,'steps':770,'unique_images':770,'all_train_covered_once':True,'mean_total':statistics.mean(x['loss_total'] for x in records),'mean_RD':statistics.mean(x['loss_rd'] for x in records),'mean_VS':statistics.mean(x['loss_vs'] for x in records),'train_seconds':end-start,'iterator_setup_ms':iterator_ms,'mean_wait_ms':statistics.mean(x['wait_ms'] for x in records),'compute_seconds':sum(x['forward_loss_ms']+x['backward_ms']+x['clip_optimizer_ms'] for x in records)/1000,'transfer_seconds':sum(x['transfer_ms'] for x in records)/1000,'peak_allocated_mib':max(x['peak_allocated_mib'] for x in records),'peak_reserved_mib':max(x['peak_reserved_mib'] for x in records),'all_gradients_finite':True,'all_parameters_updated':True,'nan_inf':False}
   train_history.append(summary);dump(EXP/f'metrics/train_epoch{epoch:03d}.json',{'summary':summary,'steps':records});print('EPOCH',json.dumps(summary),flush=True);evaluate(epoch)
   if epoch in [1,2,5,10]:save_checkpoint(epoch)
 shutdown(dl);total_seconds=time.perf_counter()-total_start
 dump(EXP/'metrics/train_history.json',{'epochs':train_history,'step_log':'train_steps.jsonl','rolling_definition':'last up to 50 global steps, crosses epoch boundaries'});dump(EXP/'metrics/dev_history.json',dev_history)
 # Fixed unranked montage: input followed by five saved epochs, identical RGB thumbnails.
 montages=[]
 for p,X,G,a in dev_inputs:
  canvas=Image.new('RGB',(6*480,300),'#20242a');draw=ImageDraw.Draw(canvas);images=[Image.fromarray(a)]
  for epoch in [0,1,2,5,10]:
   with Image.open(OUT/f'epoch{epoch:03d}'/(Path(p).stem+'.png')) as im:images.append(im.copy())
  for j,(im,label) in enumerate(zip(images,['input','epoch0','epoch1','epoch2','epoch5','epoch10'])):
   im=im.resize((480,270),Image.Resampling.LANCZOS);canvas.paste(im,(j*480,30));draw.text((j*480+8,8),Path(p).name+' | '+label,fill='white')
  target=OUT/'montage'/(Path(p).stem+'.png');target.parent.mkdir(exist_ok=True);canvas.save(target)
  with Image.open(target) as im:assert im.size==(2880,300);im.verify()
  montages.append(target.relative_to(ROOT).as_posix())
 assert len(train_history)==10 and sum(x['steps'] for x in train_history)==7700 and len(dev_history)==11 and [x['epoch'] for x in checkpoint_checks]==[1,2,5,10]
 assert all(sha(REPO/n)==h for n,h in source_hashes.items()) and all(sha(ROOT/p)==h for p,h in split_hashes.items())
 assert all(sha(ROOT/p)==h for p,h in meta['image_sha256'].items())
 final={'status':'COMPLETED_10_EPOCH_PILOT','epochs':10,'steps':7700,'training_seconds':sum(x['train_seconds'] for x in train_history),'total_training_evaluation_checkpoint_seconds':total_seconds,'total_scope':'epoch0 dev through final checkpoint verification; excludes setup, benchmark, montage and final file hash audit','checkpoint_checks':checkpoint_checks,'montages':montages,'nan_inf':False,'oom':False,'all_770_covered_each_epoch':True,'no_train_dev_quarantine_mixing':True,'source_and_input_hashes_unchanged':True,'best_checkpoint_selected':False}
 dump(EXP/'metrics/summary.json',final)
 text='# EXP005A-0: DI-Retinex-U3D 10-epoch adapted baseline pilot\n\nCompleted exactly 10 epochs, 770 images per epoch, 7700 updates. STOPPED for human research review; no extension, early stopping or best checkpoint selection.\n\nOfficial DI-Retinex dataloader uses full images. U3D originals are 3840x2160; this project uses 256x256 paired random patches after the original full-image Gaussian processing and conversion. Therefore this is an adapted baseline, not strict paper reproduction and not an innovation experiment.\n\nOriginal network and RD+VS 1:1, random official initialization, seed1149, Adam lr0.001, weight decay0.0001, clipping0.1, exposure0.5 (original Normal std0.001), batch1. No old/pretrained checkpoint loaded. All 770 train paths traversed once every epoch; dev8 and quarantine22 excluded. Loader settings and source/split/script hashes: config/run.json. Benchmark candidates, skips, equivalence and throughput: metrics/loader_benchmark.json. Only tested configurations qualify for selection; skipped RAM-heavy configurations are not claimed slower.\n\nDev0..10: full eight RGB images resized with LANCZOS to 1280x720, no crop. Eval/no_grad, parameters and training RNG unchanged; train mode restored. Raw output reconstructed and checked against official clamp. Ratios count channels. Luminance is encoded RGB Yprime, not linear-light. Unsupervised dev objective uses original RD/VS on resized inputs and their full Gaussian blur, fixed per-image exposure sample via isolated RNG for epoch comparability. It is NOT an image quality score or GT metric.\n\nEpoch0/1/2/5/10 images and eight input|epoch0|epoch1|epoch2|epoch5|epoch10 montages are Git ignored under results/demos/di_retinex_exp005a0/. No automatic preference/ranking.\n\nCheckpoints epoch001/002/005/010 under checkpoints/di_retinex/exp005a0/ are pilot-analysis-only; weights_only and strict state loading plus exact forward match verified. No best.pth.\n\nMetrics include per-epoch average total/RD/VS, global rolling50 step losses, per-step image/crop, finite gradients and parameter updates. Synchronized GPU-phase wall timing excludes separate finite/parameter-delta checks. Loader wait is main-process blocking time; with workers it is not total CPU work. Timing scopes and startup included in benchmark are explicit; do not infer whole-training speedup from loader-only throughput. Reserved CUDA memory includes cached dev allocations. No training objective adaptation or augmentation changes were made.\n\nThis small dev set and limited scene metadata cannot establish universal scene independence or final generalization. Ten epochs do not establish convergence. Human review is required before choosing any longer protocol. No commit/push.\n\n## Actual epoch summaries\n\n|Epoch|Mean total|Mean RD|Mean VS|Train seconds|Mean wait ms|\n|---|---:|---:|---:|---:|---:|\n'
 for x in train_history:text+=f"|{x['epoch']}|{x['mean_total']:.8f}|{x['mean_RD']:.8f}|{x['mean_VS']:.8f}|{x['train_seconds']:.3f}|{x['mean_wait_ms']:.3f}|\n"
 text+='\nFinal execution summary: metrics/summary.json. No best epoch conclusion.\n'
 with (EXP/'README.md').open('x',encoding='utf-8') as f:f.write(text)
 print('PILOT COMPLETE',json.dumps(final),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--benchmark',action='store_true');mode.add_argument('--train',action='store_true');a=ap.parse_args()
 try:benchmark() if a.benchmark else train()
 except Exception as e:
  if EXP.exists() and not (EXP/'failure.json').exists():dump(EXP/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'phase':'benchmark' if a.benchmark else 'train','status':'STOPPED_FAILURE_NO_RETRY'})
  raise
