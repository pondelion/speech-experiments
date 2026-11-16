"""三値QATと2bit保存。推論演算は浮動小数点で、専用kernelではない。"""
from pathlib import Path
import torch
from qat_models import layer_names

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'/'ternary'


def ternarize(weight):
    scale=weight.detach().float().abs().mean().clamp_min(1e-8)
    codes=(weight.float()/scale).round().clamp(-1,1).to(torch.int8)
    return codes,scale


def activation(x,ste):
    scale=x.detach().float().abs().amax(-1,keepdim=True).clamp_min(1e-8)/127
    value=((x.float()/scale).round().clamp(-128,127)*scale).to(x.dtype)
    return x+(value-x).detach() if ste else value


class TernaryLinear(torch.nn.Linear):
    def forward(self,x):
        codes,scale=ternarize(self.weight)
        value=(codes.float()*scale).to(self.weight.dtype)
        weight=self.weight+(value-self.weight).detach()
        return torch.nn.functional.linear(activation(x,True),weight,self.bias)


class ActivationLinear(torch.nn.Linear):
    def forward(self,x):
        return torch.nn.functional.linear(activation(x,False),self.weight,self.bias)


def qat_prepare(model):
    names=layer_names(torch,model)
    for name in names:
        old=model.get_submodule(name)
        new=TernaryLinear(old.in_features,old.out_features,old.bias is not None,device=old.weight.device,dtype=old.weight.dtype)
        new.weight=old.weight;new.bias=old.bias
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,new)
    return names


def pack(codes):
    flat=(codes.reshape(-1).to(torch.int16)+1).to(torch.uint8)
    padded=torch.nn.functional.pad(flat,(0,(-flat.numel())%4),value=1).reshape(-1,4)
    return (padded[:,0]|(padded[:,1]<<2)|(padded[:,2]<<4)|(padded[:,3]<<6)).cpu()


def unpack(packed,shape,scale,device):
    packed=packed.to(device)
    values=torch.stack([(packed>>shift)&3 for shift in [0,2,4,6]],dim=-1).flatten()
    count=1
    for n in shape:count*=n
    values=values[:count].to(torch.int8)-1
    if bool(((values<-1)|(values>1)).any()):raise RuntimeError('invalid ternary code')
    return (values.float()*scale.to(device)).reshape(shape)


def make_artifact(pipe,t2s,mode):
    from quantize import weight_only
    from benchmark import CHECKPOINTS
    RUNS.mkdir(parents=True,exist_ok=True)
    path=RUNS/f'{mode}_packed.pt'
    if path.exists():return path
    training={}
    if mode in ['ft_ptq','qat','kd']:
        source=RUNS/f'train_{"ft" if mode=="ft_ptq" else mode}.pt'
        state=torch.load(source,map_location=pipe.device,weights_only=False)
        t2s.load_state_dict(state['model']);training=state['metadata']
    names=layer_names(torch,t2s)
    packed={};checks={};meta={'mode':mode,'training':training,'activation_int8':mode not in ['baseline','int4','weight_only'],
        'quantized_layers':names if mode not in ['baseline'] else [],'packing':'four 2-bit codes per uint8; 0=-1,1=0,2=+1',
        'weight_scale':'per-tensor absmean float32','saved_high_precision':'bfloat16','kernel':'expanded floating-point F.linear'}
    if mode not in ['baseline','int4']:
        for name in names:
            module=t2s.get_submodule(name);codes,scale=ternarize(module.weight)
            bits=pack(codes)
            restored=unpack(bits,tuple(codes.shape),scale.cpu(),'cpu')
            assert torch.equal(restored,(codes.float()*scale).cpu())
            packed[name]={'packed':bits,'shape':tuple(codes.shape),'scale':scale.cpu()}
            checks[name]={'weights':codes.numel(),'packed_bytes':bits.numel(),
                         'zero_fraction':float((codes==0).float().mean()),'roundtrip_exact':True}
    t2s.to(dtype=torch.bfloat16);pipe.s2m.to(dtype=torch.bfloat16)
    if mode=='int4':meta['int4']=weight_only(torch,t2s,4,exclude={'head'},group_size=128)
    state=t2s.state_dict()
    for name in packed:del state[name+'.weight']
    original=torch.load(CHECKPOINTS['t2s'],map_location='cpu',weights_only=False)
    architecture={'t2s_config':original['config'],'vocab':original['vocab'],'t2s_method':original['method']}
    original=torch.load(CHECKPOINTS['s2m'],map_location='cpu',weights_only=False)
    architecture['s2m_config']=original['config']
    torch.save({'t2s':state,'s2m':pipe.s2m.state_dict(),'ternary':packed,'architecture':architecture,'metadata':meta},path)
    meta.update(file_bytes=path.stat().st_size,packing_checks=checks)
    out=ROOT/'results'/'ternary';out.mkdir(parents=True,exist_ok=True)
    import json
    (out/f'packing_{mode}.json').write_text(json.dumps(meta,indent=2)+'\n')
    return path


def prepare(torch,pipe,t2s,device,mode):
    path=RUNS/f'{mode}_packed.pt'
    if not path.exists():make_artifact(pipe,t2s,mode)
    saved=torch.load(path,map_location=device,weights_only=False)
    state=saved['t2s']
    for name,entry in saved['ternary'].items():
        # All deployment weights use BF16 values; CPU computes those values in FP32.
        state[name+'.weight']=unpack(entry['packed'],entry['shape'],entry['scale'],device).to(torch.bfloat16)
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    t2s.to(dtype=torch.bfloat16);pipe.s2m.to(dtype=torch.bfloat16)
    t2s.load_state_dict(state,assign=True);pipe.s2m.load_state_dict(saved['s2m'],assign=True)
    t2s.to(dtype=dtype);pipe.s2m.to(dtype=dtype)
    meta=saved['metadata']
    if meta['activation_int8']:
        for name in meta['quantized_layers']:
            old=t2s.get_submodule(name)
            new=ActivationLinear(old.in_features,old.out_features,old.bias is not None,device=device,dtype=dtype)
            new.weight=old.weight;new.bias=old.bias
            parent,_,child=name.rpartition('.')
            setattr(t2s.get_submodule(parent) if parent else t2s,child,new)
    meta={**meta,'artifact_bytes':path.stat().st_size,'runtime_dtype':str(dtype),
          'runtime_representation':'2-bit file expanded once to floating-point weights; activation quantization uses floating-point ops'}
    return meta,t2s
