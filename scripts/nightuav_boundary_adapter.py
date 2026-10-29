"""Observe the original torch.clamp input without changing the official forward."""
import torch
from torch.overrides import TorchFunctionMode

class _CaptureClamp(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.calls=[]
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        result=func(*args, **kwargs)
        if func is torch.clamp:
            self.calls.append((args,kwargs,result))
        return result

def forward_with_original_z(model,x):
    """No detach, recomputation, parameter replacement or source modification."""
    with _CaptureClamp() as observer:
        outputs=model(x)
    if len(observer.calls)!=1:
        raise RuntimeError('Expected exactly one official torch.clamp')
    args,kw,result=observer.calls[0]
    if len(args)!=3 or args[1]!=0 or args[2]!=1 or kw:
        raise RuntimeError('Unexpected clamp signature; stop instead of changing model')
    z=args[0]
    if result is not outputs[0]:
        raise RuntimeError('Captured clamp is not official returned Y')
    if torch.is_grad_enabled() and not z.requires_grad:
        raise RuntimeError('Original z is not differentiable')
    return outputs,z

def upper_loss(z):
    return torch.relu(z-1).square().mean()
