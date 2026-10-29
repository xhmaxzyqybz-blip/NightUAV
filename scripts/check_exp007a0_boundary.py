"""EXP007A-0: train-only gradient audit; deliberately no optimizer or training."""
import hashlib,json,random,sys,importlib,time,traceback,math
from pathlib import Path
import torch
from nightuav_boundary_adapter import forward_with_original_z,upper_loss
import train_di_retinex_exp005b as baseline
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/exp007a0_boundary_mechanism_check'
BASE=ROOT/'experiments/exp005b_di_retinex_u3d_baseline'
REPO=ROOT/'_model_audit/DI-retinex-low-light-enhancement'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def norm(gs):return math.sqrt(sum(float(g.detach().double().square().sum()) for g in gs))
def flatstats(gs,names):
 return {'l2':norm(gs),'finite':all(bool(torch.isfinite(g).all()) for g in gs),
         'nonzero_elements':sum(int(torch.count_nonzero(g)) for g in gs),
         'per_parameter_l2':{n:float(g.detach().double().norm()) for n,g in zip(names,gs)}}
def loss(y,b,k,x,g,fn,i):
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(1149+i);torch.cuda.manual_seed_all(1149+i)
  return fn(x,g,y,b,k,.5)

def main():
 assert not EXP.exists(),'Refuse overwrite'
 cfg=read(BASE/'config/run.json')
 protected={**cfg['protected_hashes'],**cfg['split_sha256'],
  'scripts/train_di_retinex_exp005b.py':cfg['script_sha256'],
  'scripts/nightuav_crop_adapter.py':cfg['crop_protocol']['adapter_sha256']}
 ck=ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth'
 protected[ck.relative_to(ROOT).as_posix()]='73bd2f1a50acc17aebb8c2fbc6e7b585cf8b38cc6b62573e2a2de0058a3f39d0'
 for p,h in protected.items():assert sha(ROOT/p)==h,p
 assert torch.cuda.is_available() and str(torch.__version__)==cfg['torch']
 splits={n:(ROOT/f'configs/splits/u3d_{n}_v2_1.txt').read_text().splitlines() for n in ['train','dev','quarantine']}
 assert [len(splits[n]) for n in splits]==[770,8,22]
 assert not(set(splits['train'])&set(splits['dev']) or set(splits['train'])&set(splits['quarantine']))
 meta=read(ROOT/'configs/splits/u3d_train_dev_v2_1.json')
 paths=sorted(splits['train']);random.Random(1149).shuffle(paths);paths=paths[:16]
 coords=random.Random(1149);plan=[(i,coords.randint(0,3840-256),coords.randint(0,2160-256)) for i in range(16)]
 for p in paths:assert p not in splits['dev'] and p not in splits['quarantine'] and sha(ROOT/p)==meta['image_sha256'][p]
 sys.path.insert(0,str(REPO));official=importlib.import_module('lowlight_train')
 official.set_seed(1149);torch.backends.cudnn.benchmark=False
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train()
 initial={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 params=tuple(model.parameters());names=[n for n,p in model.named_parameters()]
 fn,expressions=baseline.make_loss(torch)
 EXP.mkdir(parents=True)
 dump(EXP/'config/run.json',{'seed':1149,'samples':paths,'plan':plan,'crop_size':256,
  'selection':'sorted train v2.1 then independent Random(1149).shuffle first16; fixed before forward',
  'initialization':'fresh official weights_init seed1149; no checkpoint loaded',
  'checkpoint_hash_only':protected[ck.relative_to(ROOT).as_posix()],
  'measurement':'unclipped parameter gradient L2; 16 independent crops at identical unupdated initialization',
  'loss_expressions':expressions,'exposure':'isolated Normal(.5,.001) seeded1149+i per sample',
  'protected_hashes':protected,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),
  'script_sha256':sha(Path(__file__)),'adapter_sha256':sha(ROOT/'scripts/nightuav_boundary_adapter.py'),
  'lambda_rule_preregistered':'r * RMS(norm(grad(RD+VS))) / RMS(norm(grad(upper))); r in [.01,.05,.1]; recommend .05 only pending user confirmation',
  'optimizer_steps':0,'smoke_authorized':False})
 # Synthetic exact boundary and nonpositive controls.
 zz=torch.tensor([-.5,0.,.5,1.,1.1,2.],device='cuda',requires_grad=True)
 uu=upper_loss(zz);gg=torch.autograd.grad(uu,zz)[0]
 assert torch.equal(gg[:4],torch.zeros_like(gg[:4])) and torch.all(gg[4:]>0)
 assert float(upper_loss(zz[:4]))==0
 dump(EXP/'metrics/synthetic.json',{'z':zz.tolist(),'loss':float(uu.detach()),'gradient':gg.tolist(),'pass':True})
 ds=baseline.PatchDataset(paths);rows=[];start=time.perf_counter()
 for i,spec in enumerate(plan):
  X,G,p,xx,yy=ds[spec];x=X[None].cuda();g=G[None].cuda();assert bool(x.any())
  torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
  (y,c,b,k),z=forward_with_original_z(model,x)
  assert torch.equal(y,z.clamp(0,1)) and z.grad_fn is not None
  rd,vs,original=loss(y,b,k,x,g,fn,i);upper=upper_loss(z)
  gz=torch.autograd.grad(upper,z,retain_graph=True)[0]
  expected=2*torch.relu(z-1)/z.numel()
  assert torch.allclose(gz,expected,rtol=1e-6,atol=0)
  assert bool((gz[z<=1]==0).all())
  if bool((z>1).any()):assert bool((gz[z>1]>0).all())
  components={}
  for label,L in [('RD',rd),('VS',vs),('upper',upper),('original',original)]:
   grads=torch.autograd.grad(L,params,retain_graph=True)
   components[label]={'value':float(L.detach()),**flatstats(grads,names)}
   assert components[label]['finite']
  gzero=torch.autograd.grad(original+0.0*upper,params)
  # Reference: no observer, same model and exact exposure RNG.
  yr,cr,br,kr=model(x);rr,vv,ref=loss(yr,br,kr,x,g,fn,i)
  gr=torch.autograd.grad(ref,params)
  assert all(torch.equal(a,d) for a,d in zip([y,c,b,k],[yr,cr,br,kr]))
  assert torch.equal(original,ref) and torch.equal(rd,rr) and torch.equal(vs,vv)
  assert all(torch.equal(a,d) for a,d in zip(gzero,gr)), 'lambda0 gradient mismatch'
  torch.cuda.synchronize()
  row={'source':p,'xywh':[xx,yy,256,256],'z_gt1_count':int((z>1).sum()),'z_numel':z.numel(),
       'components':components,'z_gradient_finite':bool(torch.isfinite(gz).all()),
       'z_gradient_expected_max_abs_error':float((gz-expected).abs().max()),
       'lambda0_forward_loss_parameter_gradients_exact':True,
       'seconds':time.perf_counter()-t,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
       'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
  rows.append(row);print('GRAD PASS',i+1,p,row['z_gt1_count'],flush=True)
  del x,g,y,c,b,k,z,gz,expected,rd,vs,original,upper,gzero,yr,cr,br,kr,rr,vv,ref,gr,grads
 assert any(r['z_gt1_count'] and r['components']['upper']['nonzero_elements'] for r in rows),'No real sample covers active upper branch'
 assert all(torch.equal(v.detach().cpu(),initial[n]) for n,v in model.state_dict().items())
 for p,h in protected.items():assert sha(ROOT/p)==h
 norms={key:math.sqrt(sum(r['components'][key]['l2']**2 for r in rows)/len(rows)) for key in ['RD','VS','upper','original']}
 assert norms['upper']>0 and math.isfinite(norms['upper'])
 candidates=[{'target_gradient_ratio':r,'lambda':r*norms['original']/norms['upper']} for r in [.01,.05,.1]]
 for c in candidates:
  c['per_sample_weighted_upper_to_original']=[c['lambda']*r['components']['upper']['l2']/r['components']['original']['l2'] if r['components']['original']['l2'] else None for r in rows]
 dump(EXP/'metrics/gradients.json',rows);dump(EXP/'metrics/crop_records.json',ds.records)
 dump(EXP/'metrics/lambda_candidates.json',{'rms_unclipped_gradient_norms':norms,'candidates':candidates,'recommended_ratio':.05,'status':'AWAITING_USER_CONFIRMATION','not_optimality_claim':True})
 dump(EXP/'metrics/verification.json',{'gradient_checks':'PASS','real_active_samples':sum(r['z_gt1_count']>0 for r in rows),
  'samples':16,'lambda0_exact_all':True,'model_parameters_unchanged':True,'protected_files_unchanged':True,
  'checkpoint_loaded':False,'optimizer_steps':0,'smoke_status':'NOT_RUN_PENDING_LAMBDA_CONFIRMATION','nan_inf':False,'oom':False,'seconds':time.perf_counter()-start})
 text=['# EXP007A-0：已有普通上界机制与梯度验证','',
 '状态：梯度检查PASS；lambda待人工确认；smoke test未运行。不是原创方法，不是完整baseline训练。','',
 '## 计算图与源码对应','',
 '第三方mymodel.py的brightness_contrast原式为z=k*(x+0.5*(b-1))+0.5*(b+1)，随即torch.clamp(z,0,1)得到Y。NightUAV适配器使用TorchFunctionMode观察原torch.clamp调用，保存其输入Tensor引用，不detach、不重新计算z、不替换层、不改第三方方法。捕获返回值与官方返回Y为同一对象；若clamp次数/签名/输出身份不匹配立即停止。','',
 '新增upper=mean(relu(z-1)^2)，total=RD+VS+lambda*upper。RD/VS继续调用EXP005B的make_loss：从冻结官方lowlight_train.py提取原赋值AST，没有重写其表达式、比例、mask或Gaussian。精确表达式和源码哈希见config/run.json。','',
 '相对z的新增梯度为2*relu(z-1)/N；z<=1为0。含越界真实crop及合成边界控制均验证通过。upper通过原计算图到达共享网络参数。比较的是gradient clipping之前的L2范数；原始训练clip=0.1下的实际更新影响不等同于这些比例。','',
 '## 样本与一致性','',
 '固定train v2.1中16张：排序后Random(1149)打乱取前16，坐标独立Random(1149)固定。只使用官方train来源且不在dev/quarantine清单，原Gaussian预处理和NightUAV全黑同图重采样不变。所有测量在同一随机初始化、不更新参数的状态进行；没有加载E15或其他checkpoint。','',
 '每图lambda=0与无适配器官方forward比较：Y/c/b/k、RD、VS、total及所有参数梯度torch.equal。该检查不包含optimizer更新；本次没有optimizer.step。原模型/旧脚本/权重/split结束哈希未变。','',
 '## 梯度量级','', '|项|跨16crop梯度L2的RMS|','|---|---:|']
 for key,value in norms.items():text.append(f'|{key}|{value:.10g}|')
 text+=['','## 有限lambda规则','',
 '预先规定候选r={1%,5%,10%}；lambda=r*RMS(||grad(RD+VS)||)/RMS(||grad(upper)||)。使用全部16个固定crop，包括upper为零的样本，不按dev画质或越界严重程度筛选。','',
 '|目标RMS比例|候选lambda|','|---|---:|']
 for c in candidates:text.append(f"|{c['target_gradient_ratio']:.0%}|{c['lambda']:.10g}|")
 text+=['','建议先确认5%候选，仅作为小幅机制干预的smoke起点，不是最优lambda。该规则未考虑梯度方向、后续训练漂移、裁剪和Adam缩放，不能保证训练中每步维持该比例。lambda_candidates.json记录逐样本比例，便于查看异质性。若批准后固定该数值，不在线自适应，不依据dev反复调参。','',
 '## 等待确认与smoke协议','',
 '未运行200步或完整15轮；无新checkpoint。用户确认具体lambda后，另行重新随机初始化seed1149，最多200步，train770、batch1、256crop、Adam lr.001/wd.0001/clip.1、exposure.5、原RD:VS=1:1，继续原同图全黑重采样。不得继承本次或历史状态，也不得作为正式实验结果。','',
 '## 文件','',
 '- config/run.json：预先固定的样本/坐标、规则、环境、源码和split哈希。',
 '- metrics/synthetic.json：阈值边界与零惩罚控制。',
 '- metrics/gradients.json：每样本loss、每参数梯度范数及数值一致性。',
 '- metrics/crop_records.json：原采样及重采样记录。',
 '- metrics/lambda_candidates.json：有限候选及依据。',
 '- metrics/verification.json：PASS范围及smoke未执行状态。','',
 '没有开发集画质选择、网络/loss原项修改、训练、commit或push。']
 (EXP/'README.md').write_text('\n'.join(text)+'\n',encoding='utf8')
 print('PASS; NO TRAINING; AWAIT LAMBDA CONFIRMATION',flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  if EXP.exists() and not(EXP/'failure.json').exists():dump(EXP/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'status':'STOPPED'})
  raise
