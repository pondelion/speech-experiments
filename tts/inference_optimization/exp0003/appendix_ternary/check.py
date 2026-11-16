"""三値・2bit packing・STE・biasの最低限の整合を検査する。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from appendix_ternary.models import pack,unpack,TernaryLinear,ActivationLinear


def main():
    torch.manual_seed(0)
    codes=torch.tensor([-1,0,1,0,1,-1],dtype=torch.int8)
    assert torch.equal(unpack(pack(codes),(6,),torch.tensor(.25),'cpu'),codes.float()*.25)
    m=TernaryLinear(128,64,bias=True)
    x=torch.randn(2,5,128,requires_grad=True)
    m(x).square().mean().backward()
    assert x.grad is not None and x.grad.abs().sum()>0 and m.weight.grad.abs().sum()>0
    a=ActivationLinear(128,64,bias=True)
    with torch.no_grad():a.weight.zero_();a.bias.fill_(.3)
    assert torch.allclose(a(x.detach()),torch.full((2,5,64),.3))
    out=ROOT/'results'/'ternary';out.mkdir(parents=True,exist_ok=True)
    (out/'checks.json').write_text('{"packing_roundtrip": true, "input_gradient": true, "weight_gradient": true, "bias_unscaled": true}\n')
    print('packing/STE/bias checks: OK')


if __name__=='__main__':main()
