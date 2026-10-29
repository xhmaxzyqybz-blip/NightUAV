"""A2-2 loss-only preparation. No model forward, inference, optimizer or training."""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/nightuav_a2_2_highlight_exclusion'

def input_mask(x, *, preserve_a21_exclusions):
    x=x.detach()
    y=(x*x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
    valid=y<=.7
    if preserve_a21_exclusions:
        valid=valid & (y>=.05) & (x.amax(1)<.99)
    return valid

def structural_moments(a,x, *, preserve_a21_exclusions):
    assert a.shape==x.shape and a.ndim==4 and a.shape[1]==3
    mask=input_mask(x,preserve_a21_exclusions=preserve_a21_exclusions)
    y=(a*a.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
    logy=torch.log(y+1/255.)
    values=[];valid=[];counts=[]
    for dim in [-1,-2]:
        q=torch.tanh(torch.diff(logy,dim=dim).abs())
        m=(mask[...,1:]&mask[...,:-1]) if dim==-1 else (mask[:,1:,:]&mask[:,:-1,:])
        count=m.sum((-2,-1));ok=count>0
        for power in [1,2]:
            values.append((q.pow(power)*m).sum((-2,-1))/count.clamp_min(1))
            valid.append(ok);counts.append(count)
    return torch.stack(values,1),torch.stack(valid,1),torch.stack(counts,1)

def structural_loss(r,x,target, *, preserve_a21_exclusions):
    values,valid,counts=structural_moments(r,x,preserve_a21_exclusions=preserve_a21_exclusions)
    n=valid.sum(0);ok=n>0
    mean=(values*valid).sum(0)/n.clamp_min(1)
    loss=((mean-target.detach()).square()*ok).sum()/ok.sum().clamp_min(1)
    return loss,dict(moments=mean.detach().tolist(),valid_components=ok.tolist(),valid_edges=counts.tolist())

def checks():
    # Fixed synthetic tensors only: not outputs of a network or dataset inference.
    x=torch.tensor([[.2,.3,.8],[.4,.6,.9]],dtype=torch.float64)[None,None].repeat(1,3,1,1)
    target=torch.tensor([.05,.01,.05,.01],dtype=x.dtype,requires_grad=True)
    r=x.clone().requires_grad_()
    results=[]
    for policy in [True,False]:
        v,ok,c=structural_moments(r,x,preserve_a21_exclusions=policy)
        assert c.tolist()==[[2,2,2,2]] and ok.all()
        y=(x*x.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1)
        logy=torch.log(y+1/255.)
        qh=(logy[:,:,1]-logy[:,:,0]).abs().tanh()
        qv=(logy[:,1,:2]-logy[:,0,:2]).abs().tanh()
        expected=torch.stack([qh.mean(),qh.square().mean(),qv.mean(),qv.square().mean()])
        assert torch.allclose(v[0],expected)
        loss,info=structural_loss(r,x,target,preserve_a21_exclusions=policy)
        gr,gt=torch.autograd.grad(loss,(r,target),allow_unused=True)
        assert torch.isfinite(loss) and torch.isfinite(gr).all() and gr.abs().sum()>0 and gt is None
        assert torch.equal(gr[:,:,:,-1],torch.zeros_like(gr[:,:,:,-1]))
        # Alter only excluded predicted values: moments must stay identical.
        changed=r.detach().clone();changed[:,:,:,-1]=.01
        vv,_,_=structural_moments(changed,x,preserve_a21_exclusions=policy)
        assert torch.equal(v,vv)
        empty=x.new_full(x.shape,.9)
        z,zi=structural_loss(r,empty,target,preserve_a21_exclusions=policy)
        assert z==0 and not any(zi['valid_components'])
        gz=torch.autograd.grad(z,r)[0];assert torch.isfinite(gz).all() and gz.abs().sum()==0
        # Low valid counts: one edge is accepted, zero is invalid (same A2-1 rule).
        one=x.new_tensor([.2,.4,.9]).view(1,1,1,3).repeat(1,3,1,1)
        _,ov,oc=structural_moments(one,one,preserve_a21_exclusions=policy)
        assert oc.tolist()==[[1,1,0,0]] and ov.tolist()==[[True,True,False,False]]
        results.append(dict(preserve_a21_exclusions=policy,loss=float(loss.detach()),R_tensor_gradient_l2=float(gr.norm()),counts=info['valid_edges']))
    # Pixel eligibility around the cutoff; no model or scene-specific tuning.
    boundary=x.new_tensor([.699,.701]).view(1,1,1,2).repeat(1,3,1,1)
    assert input_mask(boundary,preserve_a21_exclusions=False).tolist()==[[[True,False]]]
    return dict(status='PASS_SYNTHETIC_ONLY',checks=results,model_forward=False,parameter_updates=0,
        verified=['adjacent endpoint masking','highlights excluded and zero R-tensor gradient there','fixed target detached','finite nonzero R-tensor gradients','empty components zero contribution','one valid edge accepted; zero invalid'],
        not_verified=['actual R-head/encoder gradients','train gradient scale','new daytime reference','initialization replay'],
        note='Both mask policies checked as unit semantics, not experimental conditions; scientific policy remains pending user clarification.')

if __name__=='__main__':
    result=checks()
    cfg=json.loads((ROOT/'experiments/nightuav_a2_1_structural_prior/config.json').read_text(encoding='utf8'))
    config=dict(status='PENDING_MASK_AND_FORWARD_CLARIFICATION',lambda_tv=cfg['lambda_tv'],lambda_struct=None,
        model='Unchanged A0',seed=1149,night_plan_source='experiments/nightuav_a2_1_structural_prior/config.json',
        initial_parameter_sha256=cfg['initial_parameter_sha256'],day_crop_source='experiments/nightuav_a2_1_structural_prior/day_reference.json',
        preserve_a21_exclusions=None,epsilon=1/255,upper_input_luma=.7,
        statistics='tanh(abs(one-pixel log-luma difference)); per-image mean(q),mean(q^2); equal valid-image mean per component',
        insufficient_count_rule='Same A2-1: count=0 excludes component; count>=1 accepted and count recorded; all invalid yields graph-connected zero; target component with no valid images must block calibration',
        calibration_rule='Pending permission: first8 fixed train batches, lambda=.1*RMS(norm(grad_rec+TV))/RMS(norm(grad_struct)); no dev or optimizer',
        training_enabled=False)
    OUT.mkdir(exist_ok=False)
    for name,value in [('config.json',config),('numeric_checks.json',result)]:
        with (OUT/name).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result))