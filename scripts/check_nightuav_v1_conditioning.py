"""Read-only E15/B0 checks. No optimizer, backward accumulation or parameter updates."""
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/src/enhancement'))
from nightuav_v1_conditioned_di import ConditionedDI
from nightuav_boundary_adapter import forward_with_original_z, upper_loss
from analyze_exp006c_float_saturation import dist, metric, COEF

def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def arr(t): return t.detach().cpu().numpy()
def error(a,b):
    d=(a-b).abs()
    return {'equal':torch.equal(a,b),'max_abs':float(d.max()),'mean_abs':float(d.mean())}
def logit(a):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(a)-np.log1p(-a)

def reachable(x,k,v):
    # Float64 arithmetic on observed float32 tensors, closed b in [0,1].
    x,k,v=[np.asarray(a,dtype=np.float64) for a in (x,k,v)]
    a=k*x+(1-k)/2; slope=(k+1)/2
    lo=np.minimum(a,a+slope); hi=np.maximum(a,a+slope)
    nonzero=slope!=0
    r0=np.divide(-a,slope,out=np.zeros_like(a),where=nonzero)
    r1=np.divide(1-a,slope,out=np.ones_like(a),where=nonzero)
    b_lo=np.maximum(0,np.minimum(r0,r1)); b_hi=np.minimum(1,np.maximum(r0,r1))
    valid=(b_lo<=b_hi)&(lo<=1)&(hi>=0)
    # Constant channels impose no delta restriction if their value is legal.
    dl=logit(np.clip(b_lo,0,1))-v; du=logit(np.clip(b_hi,0,1))-v
    dl=np.where(nonzero,dl,-np.inf); du=np.where(nonzero,du,np.inf)
    lower=dl.max(axis=1); upper=du.min(axis=1)
    shared=valid.all(axis=1)&(lower<=upper)&(lower<np.inf)&(upper>-np.inf)
    return lo,hi,shared,lower,upper

def checks(wrapper,x):
    out=wrapper(x)
    assert all(torch.isfinite(v).all() for v in out.values())
    b,v,k=out['b'],out['v'],out['k']
    vd=v.detach().double(); kd=k.detach().double(); xd=x.detach().double()
    delta=torch.zeros_like(vd[:,:1],requires_grad=True)
    _,z,_=wrapper.render(xd,vd,kd,delta)
    gradient=torch.autograd.grad(z.sum(),delta)[0]
    analytic=((kd+1)/2*torch.sigmoid(vd)*(1-torch.sigmoid(vd))).sum(1,keepdim=True)
    analytic_error=error(gradient,analytic)
    assert torch.allclose(gradient,analytic,atol=1e-10,rtol=1e-10)
    step=1e-4
    _,zp,_=wrapper.render(xd,vd,kd,delta.detach()+step)
    _,zm,_=wrapper.render(xd,vd,kd,delta.detach()-step)
    finite_difference=((zp-zm)/(2*step)).sum(1,keepdim=True)
    assert torch.allclose(finite_difference,analytic,atol=1e-7,rtol=1e-6)
    bp,zp,_=wrapper.render(xd,vd,kd,delta.detach()+.1)
    bm,zm,_=wrapper.render(xd,vd,kd,delta.detach()-.1)
    assert bool((bp>=bm).all()) and bool((zp>=zm).all()) and bool((zp>zm).any())
    params=tuple(wrapper.condition.parameters())
    def gradients(loss,retain=True):
        gs=torch.autograd.grad(loss,params,retain_graph=retain)
        assert all(torch.isfinite(g).all() for g in gs)
        return [float(g.double().norm()) for g in gs]
    g_upper=gradients(upper_loss(out['z']))
    g_pre=gradients(out['z'].mean())
    g_post=gradients(out['y'].mean())
    assert g_upper[-1]>0 and g_pre[-1]>0 and g_post[-1]>0
    assert g_upper[0]==0 and g_upper[1]==0
    # Functional nonzero fixture, never copied into stored parameters.
    fixture={n:p.detach().clone().requires_grad_(True) for n,p in wrapper.condition.named_parameters()}
    fixture['2.weight']=torch.full_like(fixture['2.weight'],.01,requires_grad=True)
    raw=torch.func.functional_call(wrapper.condition,fixture,(out['s'].detach(),))
    delta2=wrapper.delta_max*torch.tanh(raw)
    _,z2,_=wrapper.render(x,out['v'],out['k'],delta2)
    fg=torch.autograd.grad(z2.mean(),tuple(fixture.values()))
    assert all(torch.isfinite(g).all() for g in fg) and float(fg[0].norm())>0
    # Explicit clamp-vs-preclamp paths on a fixed synthetic tensor.
    test_z=torch.tensor([1.2,.4],device=x.device,requires_grad=True)
    clamp_g=torch.autograd.grad(test_z.clamp(0,1).sum(),test_z)[0]
    bound_g=torch.autograd.grad(upper_loss(test_z),test_z)[0]
    assert clamp_g[0]==0 and bound_g[0]>0
    assert all(p.grad is None for p in wrapper.parameters())
    return {'analytic_error':analytic_error,'finite_difference_error':error(finite_difference,analytic),
      'offset_fixture':[-.1,.1],'direction_pass':True,'upper_gradient_norms':g_upper,
      'preclamp_gradient_norms':g_pre,'postclamp_gradient_norms':g_post,
      'functional_nonzero_head_gradient_norms':[float(g.norm()) for g in fg],
      'stored_parameters_modified':False,'first_layer_zero_gradient_at_zero_last_layer_expected':True,
      'clamp_gradient':arr(clamp_g).tolist(),'upper_gradient':arr(bound_g).tolist()}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-delta-max',type=float,required=True,help='Numerical fixture only, not selected model hyperparameter')
    args=parser.parse_args()
    target=args.output.resolve()
    assert target.is_relative_to(ROOT/'experiments') and not target.exists()
    torch.manual_seed(1149)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    repo=ROOT/'_model_audit/DI-retinex-low-light-enhancement'
    ck=ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth'
    cfg=read(ROOT/'experiments/exp006c_float_saturation_attribution/config/run.json')
    assert sha(ck)==cfg['checkpoint_sha256']
    assert sha(repo/'mymodel.py')==cfg['protected_hashes']['_model_audit/DI-retinex-low-light-enhancement/mymodel.py']
    spec=importlib.util.spec_from_file_location('frozen_di',repo/'mymodel.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=mod.enhance_net_nopool().cuda().eval()
    model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'],strict=True)
    wrapper=ConditionedDI(model,delta_max=args.fixture_delta_max).cuda().eval()
    assert sum(p.numel() for p in wrapper.condition.parameters())==25
    before={n:p.detach().cpu().clone() for n,p in wrapper.state_dict().items()}
    fixed={r['source']:r for r in read(ROOT/'experiments/exp005b_di_retinex_u3d_baseline/config/fixed_dev_inputs.json')}
    rows=[]; numeric=None
    # Cover k<-1, k=-1 and k>-1 endpoint orientation explicitly.
    test_k=np.array([-2.,-1.,0.,2.]).reshape(1,1,1,4)
    test_x=np.full_like(test_k,.8); test_v=np.zeros_like(test_k)
    lo,hi,_,_,_=reachable(test_x,test_k,test_v)
    e0=test_k*test_x+(1-test_k)/2; e1=e0+(test_k+1)/2
    assert np.array_equal(lo,np.minimum(e0,e1)) and np.array_equal(hi,np.maximum(e0,e1))
    for index,source in enumerate(cfg['dev']):
        p=ROOT/'results/demos/di_retinex_exp005b/input'/f'{Path(source).stem}.png'
        rel=p.relative_to(ROOT).as_posix()
        assert sha(p)==cfg['protected_hashes'][rel]
        with Image.open(p) as im: a=np.array(im.convert('RGB'))
        assert tuple(a.shape)==(720,1280,3)
        assert hashlib.sha256(a.tobytes()).hexdigest()==fixed[source]['resized_rgb_sha256']
        x=torch.from_numpy(a/255.).float().permute(2,0,1).unsqueeze(0).contiguous().cuda()
        with torch.no_grad():
            original,z_ref=forward_with_original_z(model,x)
            out=wrapper(x)
            assert all(torch.isfinite(v).all() for v in out.values())
            eq={'z':error(out['z'],z_ref),'y':error(out['y'],original[0]),'b':error(out['b'],original[2])}
            assert all(e['equal'] for e in eq.values()) and bool((out['delta']==0).all())
            q=np.rint(arr(out['y'])[0].transpose(1,2,0)*255).astype('uint8')
            with Image.open(ROOT/'results/demos/di_retinex_exp005b/epoch015'/f'{Path(source).stem}.png') as im:
                assert np.array_equal(q,np.array(im.convert('RGB')))
            xx,kk,vv,zz,bb=[arr(t) for t in (x,out['k'],out['v'],out['z'],out['b'])]
            lo,hi,shared,dl,du=reachable(xx,kk,vv)
            over=zz>1; impossible=over&(lo>1)
            sensitivity=(kk.astype('float64')+1)/2*bb*(1-bb)
            luma=(xx[0].transpose(1,2,0)*COEF).sum(2)
            regions={}
            for name,mask in [('all',np.ones_like(luma,dtype=bool)),('dark',luma<.2),('middle',(luma>=.2)&(luma<=.7)),('bright',luma>.7)]:
                m=np.broadcast_to(mask[None,None],over.shape)
                n=int((over&m).sum()); count=int((impossible&m).sum())
                regions[name]={'over_channels':n,'impossible_channels':count,'impossible_over_overflow':count/n if n else None}
            op=over.any(axis=1)
            row={'source':source,'shape':list(vv.shape),'zero_offset':eq,'regions':regions,
              'k':dist(kk),'v':dist(vv),'b':dist(bb),'sigmoid_derivative':dist(bb*(1-bb)),
              'dz_ddelta':dist(sensitivity),'overflow_dz_ddelta':dist(sensitivity[over]),
              'sigmoid_derivative_below_1e-3':metric(bb*(1-bb)<1e-3),
              'dz_ddelta_below_1e-3':metric(sensitivity<1e-3),
              'overflow_dz_ddelta_below_1e-3':metric(sensitivity[over]<1e-3),
              'k_le_minus1_channels':int((kk<=-1).sum()),'limiting_zmin_equal1_overflow_channels':int((over&(lo==1)).sum()),
              'overflow_pixels':int(op.sum()),
              'overflow_pixels_without_common_finite_delta_for_all_RGB_in_0_1':int((op&~shared).sum()),
              'overflow_pixels_without_common_delta_in_fixture_bound':int((op&~(shared&(np.maximum(dl,-args.fixture_delta_max)<=np.minimum(du,args.fixture_delta_max)))).sum())}
            rows.append(row)
        if index==0:
            numeric=checks(wrapper,x)
            # Shape-only synthetic non-square input; no extra dataset sample.
            with torch.no_grad():
                non_square=wrapper(torch.zeros((1,3,33,47),device='cuda'))
                assert non_square['delta'].shape==(1,1,33,47) and non_square['y'].shape==(1,3,33,47)
        print(json.dumps({'image':Path(source).stem,**row['regions']['all']},ensure_ascii=False),flush=True)
        del x,out,original,z_ref
    assert all(torch.equal(v.cpu(),before[n]) for n,v in wrapper.state_dict().items())
    assert all(p.grad is None for p in wrapper.parameters())
    assert sha(ck)==cfg['checkpoint_sha256']
    assert sha(repo/'mymodel.py')==cfg['protected_hashes']['_model_audit/DI-retinex-low-light-enhancement/mymodel.py']
    pooled={}
    for region in rows[0]['regions']:
        n=sum(r['regions'][region]['over_channels'] for r in rows)
        count=sum(r['regions'][region]['impossible_channels'] for r in rows)
        pooled[region]={'over_channels':n,'impossible_channels':count,'impossible_over_overflow':count/n if n else None}
    total_op=sum(r['overflow_pixels'] for r in rows)
    common_bad=sum(r['overflow_pixels_without_common_finite_delta_for_all_RGB_in_0_1'] for r in rows)
    result={'status':'PASS','parameter_updates':0,'trainable_parameter_count':25,
      'fixture_delta_max':args.fixture_delta_max,'fixture_is_not_hyperparameter_selection':True,
      'k_theory_range':[math.tan(math.radians(.2)),math.tan(math.radians(89.8))],
      'definitions':{'overflow':'strict z>1 per RGB channel, all original overflow including input endpoints',
      'impossible':'closed b in [0,1] minimum z>1; denominator original overflow channels',
      'coupling':'b remains RGB; delta is shared, common interval requires all RGB z in [0,1]',
      'distribution_thresholds':'1e-3 diagnostic only, not training thresholds',
      'reachability':'float64 endpoints of observed float32 k/X; finite sigmoid endpoint limitations separate',
      'shared_feasibility':'necessary per-pixel condition; does not prove one g(S) can realize all pixels'},
      'numeric_checks':numeric,'per_image':rows,'pooled':pooled,
      'shared_delta':{'overflow_pixels':total_op,'no_common_finite_delta_pixels':common_bad,'ratio':common_bad/total_op},
      'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [repo/'mymodel.py',ck,Path(__file__),ROOT/'code/src/enhancement/nightuav_v1_conditioned_di.py']},
      'parameters_and_checkpoint_unchanged':True,'negative_k_endpoint_test_pass':True}
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8') as f: json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASS','pooled':pooled,'shared_delta':result['shared_delta'],'numeric_checks':numeric}),flush=True)

if __name__=='__main__': main()