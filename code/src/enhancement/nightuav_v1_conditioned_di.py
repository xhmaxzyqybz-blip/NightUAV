"""B0 numerical prototype; frozen DI backbone, no optimizer or training policy."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ConditionedDI(nn.Module):
    def __init__(self, backbone, *, delta_max):
        super().__init__()
        if not math.isfinite(delta_max) or delta_max <= 0:
            raise ValueError('delta_max must be explicitly specified and positive')
        if backbone.linear or backbone.e_conv3.out_channels != 6:
            raise ValueError('Only audited normal RGB DI interface is supported')
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.delta_max = float(delta_max)
        self.condition = nn.Sequential(nn.Conv2d(1, 8, 1), nn.ReLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    @staticmethod
    def render(x, v, k, delta):
        b = torch.sigmoid(v + delta)
        z = k * (x + .5 * (b - 1)) + .5 * (b + 1)
        return b, z, z.clamp(0, 1)

    def forward(self, x):
        captured = []
        def capture(module, args):
            captured.append(args[0])
        handle = self.backbone.sigmoid.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():
                original_y, c, original_b, k = self.backbone(x)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape != x.shape:
            raise RuntimeError('Unexpected sigmoid interface; refusing inferred logits')
        v = captured[0]
        luma = (x * x.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
        s = F.avg_pool2d(F.pad(luma, (15, 15, 15, 15), mode='replicate'), 31, stride=1)
        delta = self.delta_max * torch.tanh(self.condition(s))
        b, z, y = self.render(x, v, k, delta)
        return dict(y=y, z=z, b=b, delta=delta, s=s, v=v, k=k,
                    original_y=original_y, original_b=original_b)