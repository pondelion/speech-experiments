"""F: C/Eと同じT2S INT4対象にQATを適用する。"""
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def layer_names(torch,model):
    return [n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)
            and n!='head' and m.in_features%128==0 and m.out_features%8==0]


def prepare_qat(torch,model):
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig,IntxFakeQuantizeConfig
    from torchao.quantization.quant_primitives import ZeroPointDomain
    names=layer_names(torch,model)
    weight=IntxFakeQuantizeConfig(dtype=torch.uint4,group_size=128,is_symmetric=False,is_dynamic=True,
        scale_precision=torch.bfloat16,zero_point_precision=torch.bfloat16,zero_point_domain=ZeroPointDomain.FLOAT)
    quantize_(model,QATConfig(weight_config=weight,step='prepare'),filter_fn=lambda m,n:n in names)
    return names


def convert_qat(torch,model,names):
    from quantize import weight_only
    for name in names:
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model,child,model.get_submodule(name).to_linear())
    return weight_only(torch,model,4,exclude={'head'},group_size=128)


def prepare(torch,pipe,t2s,device,mode):
    from quantize import weight_only
    if device!='cuda':
        raise RuntimeError('FのINT4 Tinygemm backendはCUDA限定。CPU INT4はGで別runtimeを検討する')
    metadata={'mode':mode,'group_size':128,'trainable_model':'t2s','inference':'actual converted INT4, not fake quant'}
    if mode in ['ft','ft_ptq','qat']:
        path=ROOT/'runs'/'qat'/('qat.pt' if mode=='qat' else 'ft.pt')
        state=torch.load(path,map_location=device,weights_only=False)
        if mode=='qat':
            names=prepare_qat(torch,t2s)
        t2s.load_state_dict(state['model'])
        metadata['training']=state['metadata']
    t2s.to(dtype=torch.bfloat16)
    pipe.s2m.to(dtype=torch.bfloat16)
    if mode=='qat':
        convert_qat(torch,t2s,names)
        metadata['quantized_layers']=names
    elif mode in ['ptq','ft_ptq']:
        metadata.update(weight_only(torch,t2s,4,exclude={'head'},group_size=128))
    return metadata,t2s
