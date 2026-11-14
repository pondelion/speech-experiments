"""Eの量子化対象をモデル構造から決め、全対象・除外層を記録する。"""
from quantize import dynamic_int8, weight_only


def prepare(torch, pipe, t2s, device, mode):
    metadata = {'mode': mode, 'bits': 4 if device == 'cuda' else 8,
                'high_precision': 'bfloat16' if device == 'cuda' else 'float32',
                'selection': 'predefined structural ablation; no evaluation-set selection'}
    for label, model in [('t2s', t2s), ('s2m', pipe.s2m)]:
        if device == 'cuda':
            model.to(dtype=torch.bfloat16)
        names = {n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
        excluded = {'head'} if label == 't2s' else set()
        if mode == 'baseline' or (mode in ['t2s','s2m'] and mode != label):
            excluded |= names
        elif mode == 'boundary':
            # Keep all non-block projections and the first/last transformer blocks.
            blocks = sorted({int(n.split('.')[1]) for n in names
                             if n.startswith('blocks.') and n.split('.')[1].isdigit()})
            if not blocks:
                raise RuntimeError(f'{label}: transformer block names not found')
            excluded |= {n for n in names if not n.startswith('blocks.') or
                         int(n.split('.')[1]) in [blocks[0],blocks[-1]]}
        if device == 'cuda':
            details = weight_only(torch,model,4,exclude=excluded,group_size=128)
        else:
            converted = dynamic_int8(torch,model,exclude=excluded) if names-excluded else model
            if label == 't2s':
                t2s = converted
            else:
                pipe.s2m = converted
            details = {'quantized_layers':sorted(names-excluded),
                       'excluded_layers':sorted(names & excluded)}
        metadata[label] = details
    return metadata,t2s
