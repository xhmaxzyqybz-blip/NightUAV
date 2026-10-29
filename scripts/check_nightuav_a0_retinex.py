"""A0 random-input forward/autograd checks only; no optimizer, data or checkpoint load."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/nightuav_a0_retinex/numeric_checks.json')
    args = parser.parse_args()
    target = args.output.resolve()
    if not target.is_relative_to((ROOT/'experiments/nightuav_a0_retinex').resolve()) or target.exists():
        raise ValueError('Use a new result file within the independent A0 directory')
    import torch
    sys.path.insert(0, str(ROOT/'code/src/enhancement'))
    from nightuav_a0_retinex import RetinexA0, reconstruction_loss
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently change the tested device')
    protected = [ROOT/'_model_audit/DI-retinex-low-light-enhancement/mymodel.py',
                 ROOT/'checkpoints/di_retinex/exp005b/epoch015.pth',
                 ROOT/'experiments/nightuav_v1_b1_probe/checkpoints/attempt500.pth',
                 ROOT/'configs/splits/u3d_train_v2_1.txt', ROOT/'configs/splits/u3d_dev_v2_1.txt',
                 ROOT/'configs/splits/u3d_quarantine_v2_1.txt']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    torch.manual_seed(1149); torch.cuda.manual_seed_all(1149)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    model = RetinexA0().cuda()
    before = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    named = list(model.named_parameters()); params = tuple(p for _,p in named)
    assert sum(p.numel() for p in params) == 10276
    rows = []; start = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for case,shape in [('random_patch',(1,3,256,256)),('random_odd_batch',(2,3,33,47)),
                       ('black',(1,3,32,32)),('white',(1,3,32,32))]:
        x = torch.rand(shape,device='cuda') if case.startswith('random') else torch.full(shape,0. if case=='black' else 1.,device='cuda')
        out = model(x)
        assert out['I'].shape == (shape[0],1,shape[2],shape[3])
        assert out['R'].shape == out['X_hat'].shape == x.shape
        assert all(bool(torch.isfinite(v).all()) and bool(((v>=0)&(v<=1)).all()) for v in out.values())
        assert torch.equal(out['X_hat'],out['I']*out['R'])
        loss = reconstruction_loss(out['X_hat'],x)
        expected = (out['X_hat']-x).abs().sum()/x.numel()
        assert torch.equal(loss,expected) and bool(torch.isfinite(loss))
        gi,gr = torch.autograd.grad(loss,(out['I'],out['R']),retain_graph=True)
        assert bool(torch.isfinite(gi).all()) and bool(torch.isfinite(gr).all())
        assert float(gi.norm())>0 and float(gr.norm())>0
        grads = torch.autograd.grad(loss,params)
        assert all(bool(torch.isfinite(g).all()) for g in grads)
        groups = {prefix:sum(float(g.double().square().sum()) for (name,_),g in zip(named,grads) if name.startswith(prefix))**.5
                  for prefix in ['encoder','illumination_head','reflectance_head']}
        assert groups['illumination_head']>0 and groups['reflectance_head']>0
        if case.startswith('random'): assert groups['encoder']>0
        rows.append(dict(case=case,input_shape=list(shape),output_shapes={k:list(v.shape) for k,v in out.items()},
            ranges={k:[float(v.detach().min()),float(v.detach().max())] for k,v in out.items()},
            loss=float(loss.detach()),loss_formula_abs_error=float((loss-expected).detach().abs()),
            output_gradient_norms={'I':float(gi.norm()),'R':float(gr.norm())},parameter_gradient_norms=groups,
            finite=True,product_exact=True))
        del x,out,loss,expected,gi,gr,grads
    # Analytic test independent of the network: I=.5, R=.4, X=.7.
    i=torch.full((1,1,1,1),.5,dtype=torch.float64,device='cuda',requires_grad=True)
    r=torch.full((1,3,1,1),.4,dtype=torch.float64,device='cuda',requires_grad=True)
    x=torch.full_like(r,.7)
    loss=reconstruction_loss(i*r,x);gi,gr=torch.autograd.grad(loss,(i,r))
    assert abs(float(loss.detach())-.5)<1e-12
    assert torch.allclose(gi,torch.full_like(i,-.4),atol=1e-12,rtol=0)
    assert torch.allclose(gr,torch.full_like(r,-1/6),atol=1e-12,rtol=0)
    assert all(torch.equal(t.detach().cpu(),before[n]) for n,t in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    for p,h in hashes.items(): assert sha(ROOT/p)==h,p
    torch.cuda.synchronize()
    record=dict(status='PASS',seed=1149,device=torch.cuda.get_device_name(0),torch=str(torch.__version__),
        parameter_count=10276,initialization='encoder Kaiming normal ReLU; heads Xavier uniform gain .1; all biases zero',
        domain='encoded RGB in [0,1]; not calibrated physical radiance',cases=rows,
        analytic_loss_and_gradients_pass=True,parameter_updates=0,parameters_unchanged=True,
        dataset_used=False,checkpoint_loaded=False,protected_hashes_unchanged=True,protected_sha256=hashes,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,elapsed_seconds=time.perf_counter()-start,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'code/src/enhancement/nightuav_a0_retinex.py']})
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x',encoding='utf8') as f: json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__': main()