"""20-step CUDA smoke test: official model, initialization and AST-extracted losses.
No validation; original U3D train only. Original loader then synchronized random crop.
"""
import ast,hashlib,importlib,importlib.metadata,json,os,random,statistics,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/exp003_di_retinex_smoke';RUN=EXP/'training';CK=ROOT/'checkpoints/di_retinex/exp003_smoke';COMMIT='3f94f95ffb181904a837532a70adc064bee793a2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
 with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False)
def main():
 import torch,cv2,natsort,numpy as np
 from torch.distributions.normal import Normal
 assert Path(sys.prefix).name.lower()=='nightuav' and sys.version_info[:2]==(3,10), 'Use the nightuav Python 3.10 environment'
 assert str(torch.__version__)=='2.13.0+cu130';assert importlib.metadata.version('torchvision')=='0.28.0+cu130';assert torch.cuda.is_available()
 repo=ROOT/'_model_audit/DI-retinex-low-light-enhancement'
 def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
 assert git('rev-parse','HEAD')==COMMIT
 for name in ['mymodel.py','lowlight_train.py','dataloader.py']:assert git('hash-object',str(repo/name))==git('rev-parse','HEAD:'+name)
 rows=json.loads((EXP/'audit/smoke_candidates.json').read_text());rows=[r for r in rows if Path(r['source']).stem=='0102']
 assert len(rows)==1
 for row in rows:
  p=(ROOT/row['source']).resolve();assert p.is_relative_to((ROOT/'datasets/U3D/lowlight/train').resolve());assert sha(p)==row['sha256']
 assert not RUN.exists() and not CK.exists(),'Refuse overwrite'
 RUN.mkdir();CK.mkdir(parents=True)
 sys.path.insert(0,str(repo));official=importlib.import_module('lowlight_train')
 # Import required cv2/natsort via actual original dataloader; no CLI training call.
 official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 crop_rng=random.Random(1149)
 net=official.mymodel.enhance_net_nopool().cuda();net.apply(official.weights_init);net.train()
 dataset=official.dataloader.lowlight_loader(str(ROOT/'datasets/U3D/lowlight/train'))
 dataset.train_list=[str(ROOT/r['source']) for r in rows];dataset.data_list=dataset.train_list
 optimizer=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
 # Use the actual upstream AST expressions, not a rewritten loss approximation.
 tree=ast.parse((repo/'lowlight_train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 required=['mask','Gaussian','target_exposure','ratio','loss1','loss2','loss']
 assignments={n.targets[0].id:n for n in ast.walk(fn) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in required}
 assert set(assignments)==set(required)
 body=[assignments[k] for k in required]+[ast.Return(value=ast.Tuple(elts=[ast.Name(id=k,ctx=ast.Load()) for k in ['loss1','loss2','loss','target_exposure']],ctx=ast.Load()))]
 args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ['img_lowlight','img_gauss','enhanced_image','b','k','exp']],kwonlyargs=[],kw_defaults=[],defaults=[])
 fun=ast.FunctionDef(name='official_losses',args=args,body=body,decorator_list=[]);module=ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[]));space={'torch':torch,'Normal':Normal};exec(compile(module,str(repo/'lowlight_train.py'),'exec'),space);loss_fn=space['official_losses']
 config=dict(seed=1149,steps=20,batch_size=1,crop=[256,256],random_crop='uniform x/y; original-resolution full-image GaussianBlur before paired crop',num_workers=0,optimizer='Adam',lr=.001,betas=[.9,.999],eps=1e-8,weight_decay=.0001,grad_clip=.1,loss_weights=[1,1],exposure_mean=.5,exposure_std=.001,precision='FP32',initialization='official weights_init; random not pretrained',model_commit=COMMIT,source_hashes={n:sha(repo/n) for n in ['mymodel.py','lowlight_train.py','dataloader.py']},wrapper_sha256=sha(Path(__file__)),data=rows,validation=False,torch=str(torch.__version__),torchvision=importlib.metadata.version('torchvision'),cuda=torch.version.cuda,cv2=cv2.__version__,opencv_distribution=importlib.metadata.version('opencv-python-headless'),natsort=natsort.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0),checkpoint=str(CK/'step020.pth'),loss_ast_source={k:ast.unparse(assignments[k]) for k in required})
 dump(RUN/'config.json',config)
 initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()};records=[]
 with (RUN/'steps.jsonl').open('x',encoding='utf-8') as log:
  for step in range(20):
   t=time.perf_counter();X,G,path=dataset[step%len(dataset)];h,w=X.shape[1:];xx=crop_rng.randint(0,w-256);yy=crop_rng.randint(0,h-256)
   X=X[:,yy:yy+256,xx:xx+256].unsqueeze(0).contiguous().cuda();G=G[:,yy:yy+256,xx:xx+256].unsqueeze(0).contiguous().cuda()
   assert X.shape==(1,3,256,256) and 0<=float(X.min())<=float(X.max())<=1 and float(X.mean())>0
   assert torch.isfinite(G).all();torch.cuda.synchronize();load_ms=(time.perf_counter()-t)*1000
   before=[p.detach().clone() for p in net.parameters()];torch.cuda.reset_peak_memory_stats();start_ev=torch.cuda.Event(enable_timing=True);end_ev=torch.cuda.Event(enable_timing=True)
   torch.cuda.synchronize();t=time.perf_counter();start_ev.record()
   Y,c,b,k=net(X);l1,l2,loss,E=loss_fn(X,G,Y,b,k,.5)
   assert all(torch.isfinite(z).all().item() for z in [Y,c,b,k,l1,l2,loss]),'Nonfinite forward/loss'
   optimizer.zero_grad();loss.backward()
   assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in net.parameters()),'Invalid gradients'
   grad=torch.nn.utils.clip_grad_norm_(net.parameters(),.1,error_if_nonfinite=True);optimizer.step();end_ev.record();torch.cuda.synchronize();ms=(time.perf_counter()-t)*1000
   delta=sum(float((p.detach()-v).abs().sum()) for p,v in zip(net.parameters(),before));assert delta>0,'No parameter update';assert all(torch.isfinite(p).all() for p in net.parameters())
   rec=dict(step=step+1,source=Path(path).relative_to(ROOT).as_posix(),crop_xywh=[xx,yy,256,256],loss_rd=float(l1),loss_vs=float(l2),loss_total=float(loss),target_exposure=float(E),gradient_norm_before_clip=float(grad),parameter_delta_l1=delta,forward_backward_update_ok=True,synchronized_step_ms=ms,cuda_event_ms=start_ev.elapsed_time(end_ev),load_transfer_ms=load_ms,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,saturated_channels=float((Y>=1).float().mean()))
   log.write(json.dumps(rec)+'\n');log.flush();records.append(rec);print(json.dumps(rec),flush=True)
   del X,G,Y,c,b,k,l1,l2,loss,before
 checkpoint={'model':{k:v.detach().cpu() for k,v in net.state_dict().items()},'optimizer':optimizer.state_dict(),'step':20,'seed':1149,'source_commit':COMMIT,'torch_rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state()}
 with (CK/'step020.pth').open('xb') as f:torch.save(checkpoint,f)
 loaded=torch.load(CK/'step020.pth',map_location='cpu',weights_only=True);assert loaded['step']==20
 assert all(torch.equal(loaded['model'][k],v.detach().cpu()) for k,v in net.state_dict().items())
 assert any(not torch.equal(initial[k],v) for k,v in loaded['model'].items())
 for row in rows:assert sha(ROOT/row['source'])==row['sha256']
 summary=dict(success=True,steps=20,first_step_ms=records[0]['synchronized_step_ms'],remaining_19_mean_ms=statistics.mean(r['synchronized_step_ms'] for r in records[1:]),remaining_19_median_ms=statistics.median(r['synchronized_step_ms'] for r in records[1:]),peak_allocated_mib=max(r['peak_allocated_mib'] for r in records),peak_reserved_mib=max(r['peak_reserved_mib'] for r in records),checkpoint=str(CK/'step020.pth'),checkpoint_sha256=sha(CK/'step020.pth'),checkpoint_safe_reload=True,original_input_hash_unchanged=True,validation_run=False,timing_scope='forward+loss+backward+gradient checks+clip+Adam; CUDA synchronized; includes diagnostic sync overhead, excludes loading/transfer/checkpoint and parameter-delta check')
 dump(RUN/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if RUN.exists() and not (RUN/'failure.json').exists():dump(RUN/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
  raise
