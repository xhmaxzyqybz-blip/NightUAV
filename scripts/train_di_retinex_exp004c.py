"""EXP004C: 200-step train/checkpoint/dev integration only. No checkpoint initialization."""
import ast,hashlib,importlib,importlib.metadata,json,random,subprocess,sys,time,traceback,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/exp004c_train_dev_smoke'
OUT=ROOT/'results/demos/di_retinex_exp004c'
CK=ROOT/'checkpoints/di_retinex/exp004c'
COMMIT='3f94f95ffb181904a837532a70adc064bee793a2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)
def single(batch):
 assert len(batch)==1
 return batch[0]
def main():
 import torch,numpy as np,cv2
 from PIL import Image
 from torch.distributions.normal import Normal
 assert Path(sys.executable).resolve()==Path('E:/anaconda3/envs/nightuav/python.exe').resolve()
 assert torch.cuda.is_available() and str(torch.__version__)=='2.13.0+cu130'
 assert importlib.metadata.version('torchvision')=='0.28.0+cu130'
 splitfiles={n:ROOT/f'configs/splits/u3d_{n}_v2_1.txt' for n in ['train','dev','quarantine']}
 splits={n:p.read_text(encoding='utf-8').splitlines() for n,p in splitfiles.items()}
 for n,count in [('train',770),('dev',8),('quarantine',22)]:
  assert len(splits[n])==len(set(splits[n]))==count
  assert all((ROOT/p).is_file() and (ROOT/p).resolve().is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve()) for p in splits[n])
 assert all(not set(splits[a])&set(splits[b]) for a,b in [('train','dev'),('train','quarantine'),('dev','quarantine')])
 splitmeta=ROOT/'configs/splits/u3d_train_dev_v2_1.json';meta=json.loads(splitmeta.read_text(encoding='utf-8'))
 inputs={p:sha(ROOT/p) for n in splits for p in splits[n]};assert inputs==meta['image_sha256']
 repo=ROOT/'_model_audit/DI-retinex-low-light-enhancement'
 def git(*a):return subprocess.check_output(['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo),*a],text=True).strip()
 assert git('rev-parse','HEAD')==COMMIT
 source_hashes={n:sha(repo/n) for n in ['mymodel.py','dataloader.py','lowlight_train.py']}
 for n in source_hashes:assert git('hash-object',str(repo/n))==git('rev-parse','HEAD:'+n)
 assert not any(p.exists() for p in [EXP,OUT,CK]),'Refuse overwrite'
 ignore=subprocess.run(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT),'check-ignore','-q','results/demos/di_retinex_exp004c/step000/probe.png'])
 assert ignore.returncode==0,'Output images must be ignored'
 sys.path.insert(0,str(repo));official=importlib.import_module('lowlight_train')
 official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train()
 initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
 # Compile the exact original loss expressions without rewriting their objectives.
 tree=ast.parse((repo/'lowlight_train.py').read_text(encoding='utf-8'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 keys=['mask','Gaussian','target_exposure','ratio','loss1','loss2','loss']
 assignments={n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in keys};assert set(assignments)==set(keys)
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['img_lowlight','img_gauss','enhanced_image','b','k','exp']],kwonlyargs=[],kw_defaults=[],defaults=[])
 body=[assignments[k] for k in keys]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['loss1','loss2','loss']],ctx=ast.Load()))]
 fun=ast.FunctionDef(name='loss_fn',args=args,body=body,decorator_list=[]);space={'torch':torch,'Normal':Normal};exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),str(repo/'lowlight_train.py'),'exec'),space);loss_fn=space['loss_fn']
 dataset=official.dataloader.lowlight_loader(str(ROOT/'datasets/U3D/lowlight/train'))
 dataset.train_list=[str(ROOT/p) for p in splits['train']];dataset.data_list=dataset.train_list
 assert {Path(p).relative_to(ROOT).as_posix() for p in dataset.data_list}==set(splits['train'])
 loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True,num_workers=0,pin_memory=False,collate_fn=single,generator=torch.Generator().manual_seed(1149))
 crop_rng=random.Random(1149)
 for p in [EXP/'config',EXP/'metrics',OUT,CK]:p.mkdir(parents=True)
 config={'purpose':'train_dev_integration_only_not_baseline_or_paper_reproduction','checkpoint_reuse_for_formal_training':False,'seed':1149,'sampler_seed':1149,'crop_seed':1149,'initialization':'random official weights_init; no checkpoint loaded','steps':200,'batch_size':1,'train_crop':[256,256],'train_preprocessing':'official full-resolution RGB decode, full-image GaussianBlur(9,9,sigma=1.7), float64 division then float32, paired uniform crop','num_workers':0,'sampler':'random permutation of all 770 train files; without replacement','optimizer':'Adam','lr':.001,'weight_decay':.0001,'betas':[.9,.999],'eps':1e-8,'grad_clip_norm':.1,'loss_weights':{'RD':1,'VS':1},'exposure_mean':.5,'exposure_std':.001,'loss_ast':{k:ast.unparse(assignments[k]) for k in keys},'dev_steps':[0,50,100,150,200],'dev_size':[1280,720],'dev_resize':'Pillow LANCZOS RGB full image; no random crop','dev_metrics':'encoded RGB luminance Yprime=.2126R+.7152G+.0722B; not linear-light or GT quality','dev_objective':'not computed; diagnostic statistics only','precision':'FP32','tf32':False,'cudnn_benchmark':False,'split_counts':{k:len(v) for k,v in splits.items()},'split_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [*splitfiles.values(),splitmeta]},'model_commit':COMMIT,'source_sha256':source_hashes,'script_sha256':sha(Path(__file__)),'python':sys.version,'torch':str(torch.__version__),'torchvision':importlib.metadata.version('torchvision'),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'checkpoint':(CK/'step200.pth').relative_to(ROOT).as_posix(),'command':[sys.executable,'-B',str(Path(__file__))]}
 dump(EXP/'config/run.json',config)
 dev_history=[];train_history=[];saved=[]
 def load_dev(p):
  with Image.open(ROOT/p) as im:
   assert im.size==(3840,2160)
   a=np.array(im.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS),dtype=np.float32)/255.
  return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).contiguous().cuda()
 def evaluate(step):
  model.eval();snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone();result=[];start=time.perf_counter()
  try:
   with torch.no_grad():
    for source in splits['dev']:
     x=load_dev(source);assert x.device.type=='cuda';torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
     y,c,b,k=model(x);torch.cuda.synchronize();forward_ms=(time.perf_counter()-t)*1000
     z=k*(x+.5*(b-1))+.5*(b+1)
     assert all(torch.isfinite(v).all().item() for v in [x,y,c,b,k,z]),'Nonfinite dev'
     assert torch.equal(z.clamp(0,1),y),'Preclamp reconstruction mismatch'
     luma=(y[0]*torch.tensor([.2126,.7152,.0722],device='cuda')[:,None,None]).sum(0);q=torch.quantile(luma.flatten(),torch.tensor([.5,.95],device='cuda'))
     rec={'step':step,'source':source,'raw_min':float(z.min()),'raw_max':float(z.max()),'nan_count':int(torch.isnan(z).sum()),'inf_count':int(torch.isinf(z).sum()),'raw_lt0_channel_fraction':float((z<0).float().mean()),'raw_gt1_channel_fraction':float((z>1).float().mean()),'clamped_eq0_channel_fraction':float((y==0).float().mean()),'clamped_eq1_channel_fraction':float((y==1).float().mean()),'mean_luminance':float(luma.mean()),'median_luminance':float(q[0]),'p95_luminance':float(q[1]),'forward_ms':forward_ms,'cuda':True,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
     if step in [0,200]:
      target=OUT/f'step{step:03d}'/(Path(source).stem+'.png');assert target.resolve()!= (ROOT/source).resolve();target.parent.mkdir(exist_ok=True)
      pixels=np.rint(y[0].permute(1,2,0).cpu().numpy()*255).astype('uint8')
      with target.open('xb') as f:Image.fromarray(pixels).save(f,format='PNG')
      with Image.open(target) as im:assert im.size==(1280,720) and im.mode=='RGB';im.load()
      saved.append(target.relative_to(ROOT).as_posix())
     result.append(rec);del x,y,c,b,k,z,luma,q
   assert all(torch.equal(snapshot[k],v.detach().cpu()) for k,v in model.state_dict().items()),'Dev modified parameters'
   assert torch.equal(rng,torch.get_rng_state()) and torch.equal(crng,torch.cuda.get_rng_state()),'Dev changed training RNG'
  finally:model.train()
  dev_history.append({'step':step,'images':result,'evaluation_wall_seconds':time.perf_counter()-start,'no_parameter_update':True,'no_rng_change':True,'train_mode_restored':model.training})
  dump(EXP/f'metrics/dev_step{step:03d}.json',dev_history[-1]);print('DEV',step,'PASS',flush=True)
 evaluate(0)
 iterator=iter(loader);overall=time.perf_counter()
 with (EXP/'metrics/train_steps.jsonl').open('x',encoding='utf-8') as log:
  for step in range(1,201):
   t=time.perf_counter();X,G,path=next(iterator);source=Path(path).relative_to(ROOT).as_posix();assert source in splits['train'] and source not in splits['dev'] and source not in splits['quarantine']
   h,w=X.shape[1:];xx=crop_rng.randint(0,w-256);yy=crop_rng.randint(0,h-256);X=X[:,yy:yy+256,xx:xx+256].unsqueeze(0).contiguous();G=G[:,yy:yy+256,xx:xx+256].unsqueeze(0).contiguous();data_ms=(time.perf_counter()-t)*1000
   torch.cuda.synchronize();t=time.perf_counter();X=X.cuda();G=G.cuda();torch.cuda.synchronize();transfer_ms=(time.perf_counter()-t)*1000
   assert X.device.type=='cuda' and torch.isfinite(X).all() and torch.isfinite(G).all() and X.mean()>0
   before=[p.detach().clone() for p in model.parameters()];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();y,c,b,k=model(X);rd,vs,loss=loss_fn(X,G,y,b,k,.5);torch.cuda.synchronize();forward_ms=(time.perf_counter()-t)*1000
   assert all(torch.isfinite(v).all().item() for v in [y,c,b,k,rd,vs,loss])
   optimizer.zero_grad();t=time.perf_counter();loss.backward();torch.cuda.synchronize();backward_ms=(time.perf_counter()-t)*1000
   assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
   t=time.perf_counter();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();update_ms=(time.perf_counter()-t)*1000
   delta=sum(float((p.detach()-v).abs().sum()) for p,v in zip(model.parameters(),before));assert delta>0 and all(torch.isfinite(p).all() for p in model.parameters())
   row={'step':step,'source':source,'crop_xywh':[xx,yy,256,256],'loss_total':float(loss.detach()),'loss_rd':float(rd.detach()),'loss_vs':float(vs.detach()),'gradient_finite':True,'gradient_norm_before_clip':float(grad),'parameter_updated':True,'parameter_delta_l1':delta,'cuda':True,'data_preprocess_ms':data_ms,'transfer_ms':transfer_ms,'forward_loss_ms':forward_ms,'backward_ms':backward_ms,'clip_optimizer_ms':update_ms,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
   train_history.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
   if step%10==0:print('TRAIN',step,'loss',row['loss_total'],flush=True)
   del X,G,y,c,b,k,rd,vs,loss,before
   if step%50==0:evaluate(step)
 loop_seconds=time.perf_counter()-overall
 dump(EXP/'metrics/train_history.json',train_history);dump(EXP/'metrics/dev_history.json',dev_history)
 checkpoint={'purpose':'EXP004C_integration_only_NEVER_reuse_for_formal_baseline','model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'step':200,'seed':1149,'source_commit':COMMIT}
 with (CK/'step200.pth').open('xb') as f:torch.save(checkpoint,f)
 loaded=torch.load(CK/'step200.pth',map_location='cpu',weights_only=True);fresh=official.mymodel.enhance_net_nopool();fresh.load_state_dict(loaded['model'],strict=True)
 assert all(torch.equal(v,checkpoint['model'][k]) for k,v in fresh.state_dict().items());assert any(not torch.equal(v,initial[k]) for k,v in fresh.state_dict().items())
 fresh=fresh.cuda().eval();model.eval()
 with torch.no_grad():
  x=load_dev(splits['dev'][0]);a=fresh(x)[0];b=model(x)[0];torch.cuda.synchronize();assert torch.isfinite(a).all() and torch.equal(a,b)
 model.train();del fresh,x,a,b
 assert len(train_history)==200 and len({x['source'] for x in train_history})==200
 assert [x['step'] for x in dev_history]==[0,50,100,150,200] and all(len(x['images'])==8 for x in dev_history)
 assert len(saved)==16
 assert all(sha(ROOT/p)==h for p,h in inputs.items())
 assert all(sha(repo/n)==h for n,h in source_hashes.items())
 assert all(sha(ROOT/p)==h for p,h in config['split_hashes'].items())
 summary={'status':'EXP004C: PASS','steps':200,'unique_train_images':200,'dev_evaluations':5,'dev_forwards':40,'saved_images':saved,'checkpoint_safe_strict_reload':True,'reload_forward_exactly_matches':True,'checkpoint_sha256':sha(CK/'step200.pth'),'train_dev_mixing':False,'nan_inf':False,'oom':False,'input_and_source_hashes_unchanged':True,'loop_seconds_including_dev50_200':loop_seconds,'train_compute_seconds':sum(x['forward_loss_ms']+x['backward_ms']+x['clip_optimizer_ms'] for x in train_history)/1000,'train_data_seconds':sum(x['data_preprocess_ms'] for x in train_history)/1000,'mean_data_preprocess_ms':statistics.mean(x['data_preprocess_ms'] for x in train_history),'mean_transfer_ms':statistics.mean(x['transfer_ms'] for x in train_history),'first_train_compute_ms':sum(train_history[0][k] for k in ['forward_loss_ms','backward_ms','clip_optimizer_ms']),'remaining_mean_compute_ms':statistics.mean(sum(x[k] for k in ['forward_loss_ms','backward_ms','clip_optimizer_ms']) for x in train_history[1:]),'train_peak_allocated_mib':max(x['peak_allocated_mib'] for x in train_history),'train_peak_reserved_mib':max(x['peak_reserved_mib'] for x in train_history),'dev_peak_allocated_mib':max(x['peak_allocated_mib'] for e in dev_history for x in e['images']),'dev_peak_reserved_mib':max(x['peak_reserved_mib'] for e in dev_history for x in e['images']),'loss_first':train_history[0]['loss_total'],'loss_last':train_history[-1]['loss_total'],'loss_first20_mean':statistics.mean(x['loss_total'] for x in train_history[:20]),'loss_last20_mean':statistics.mean(x['loss_total'] for x in train_history[-20:])}
 dump(EXP/'metrics/summary.json',summary)
 readme='# EXP004C train/dev integration smoke test\n\n'+json.dumps(summary,indent=2)+'\n\nU3D v2.1: train 770, dev 8, quarantine 22. 200 steps, batch 1, train crop 256x256, random initialization seed 1149. Official network and AST-extracted RD:VS=1:1 unchanged. Adam lr .001, wd .0001, clip .1. Full-image original loader preprocessing precedes paired random crop; no caching optimization. All 770 entries are eligible; a shuffled permutation supplies 200 different source images.\n\nDev is evaluation only at 0/50/100/150/200, eval plus no_grad, no backward or optimizer calls, no crop or random operations; model parameters and CPU/CUDA RNG verified unchanged, then train mode restored. Full dev images are RGB LANCZOS resized to 1280x720, NOT native 4K evaluation. Raw preclamp expression reconstructed from official returned k/b and verified against clamped official output. Channel fractions are not pixel fractions. Luminance statistics are encoded RGB Yprime, not linear-light. No GT, PSNR, SSIM, or image-quality score. No unsupervised dev objective computed.\n\nOnly step000 and step200 images saved (8 each); Git ignored. Checkpoint: checkpoints/di_retinex/exp004c/step200.pth, independent integration-only storage. Rebuilt fresh model, weights_only=True, strict=True, exact state and CUDA dev output match. NEVER inherit this checkpoint for formal baseline training: reinitialize randomly.\n\nTiming: synchronized wall times for forward+loss, backward, clip+optimizer separately; diagnostic finite checks/parameter differences are outside these times. Data timing covers full decode, blur, conversion and crop; GPU transfer is separate. Loader workers=0. Train reserved memory can include cache retained from preceding dev evaluation; allocator peaks are not process-wide GPU usage. First training step is separate from remaining steps. Loop time includes four in-loop dev evaluations and logging, excludes initial step0 evaluation, startup and checkpoint verification.\n\nThis is NOT formal baseline training, NOT full paper reproduction, NOT an innovation experiment and cannot support final generalization conclusions. Official val/test and quarantine never enter loaders. Current dev split remains limited by 8 representatives and incomplete scene metadata. Technical integration success does not determine full training schedule, early stopping or convergence.\n\nReproduction command and exact source/split/script hashes: config/run.json. Per-step samples, losses, gradients, update checks and timing: metrics/train_history.json; dev raw ranges and luminance: metrics/dev_history.json. No commit or push.\n'
 with (EXP/'README.md').open('x',encoding='utf-8') as f:f.write(readme)
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if EXP.exists():
   if not (EXP/'failure.json').exists():dump(EXP/'failure.json',{'status':'FAIL','error':repr(e),'traceback':traceback.format_exc()})
   if not (EXP/'README.md').exists():(EXP/'README.md').write_text('# EXP004C FAILED\n\nSee failure.json and partial metrics. No automatic retry.\n',encoding='utf-8')
  raise
