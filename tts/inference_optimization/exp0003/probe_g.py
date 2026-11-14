"""可変長・padding付き入力でONNX出力と実行providerを確認する。"""
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'exp0001'))
from benchmark import build_models
from onnx_models import quantized_paths


def main():
    import numpy as np
    import onnxruntime as ort
    torch,pipe,t2s,*_=build_models('cpu');torch.set_num_threads(8)
    ort.preload_dlls()
    result=[]
    for device,mode in [('cpu','fp32'),('cpu','int8'),('cpu','int4'),('cuda','fp32'),('cuda','int4')]:
        record={'device':device,'mode':mode,'graphs':{}}
        for name,path in zip(['encoder','blocks','velocity'],quantized_paths(mode)):
            options=ort.SessionOptions();options.intra_op_num_threads=8;options.enable_profiling=True
            options.profile_file_prefix=str(ROOT/'runs'/'onnx'/f'profile_{device}_{mode}_{name}')
            providers=['CPUExecutionProvider'] if device=='cpu' else [('CUDAExecutionProvider',{'use_tf32':0}),'CPUExecutionProvider']
            session=ort.InferenceSession(str(path),sess_options=options,providers=providers);session.disable_fallback()
            if device=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():raise RuntimeError('CUDA provider inactive')
            errors=[]
            for batch,length in [(1,8),(2,32),(4,65)]:
                torch.manual_seed(0)
                x=torch.randn(batch,length,512);pad=torch.zeros(batch,length,dtype=torch.bool);pad[:,-3:]=True
                with torch.inference_mode():
                    if name=='encoder':
                        args=[x,(~pad).long()];expected=t2s.encoder(inputs_embeds=args[0],attention_mask=args[1]).last_hidden_state
                    elif name=='blocks':
                        args=[x,torch.randn(batch,512),pad];expected=x
                        for block in t2s.blocks:expected=block(expected,args[1],pad)
                    else:
                        args=[torch.randn(batch,length,80),x,torch.rand(batch),pad];expected=pipe.s2m.velocity(*args)
                actual=session.run(None,{spec.name:v.numpy() for spec,v in zip(session.get_inputs(),args)})[0]
                assert np.isfinite(actual).all()
                relative=float(np.linalg.norm(actual-expected.numpy())/np.linalg.norm(expected.numpy()))
                if mode=='fp32' and relative>.01:raise RuntimeError(f'FP32 numerical mismatch {name}: {relative}')
                errors.append({'batch':batch,'length':length,'relative_l2':relative})
            profile=json.loads(Path(session.end_profiling()).read_text())
            placement=Counter((x.get('args',{}).get('op_name'),x.get('args',{}).get('provider')) for x in profile if 'provider' in x.get('args',{}))
            cpu_heavy=[op for op,provider in placement if provider=='CPUExecutionProvider' and op in ['MatMul','Gemm','MatMulNBits','MatMulInteger','Attention','MultiHeadAttention']]
            if device=='cuda' and cpu_heavy:raise RuntimeError(f'GPU heavy-op CPU fallback: {cpu_heavy}')
            record['graphs'][name]={'errors':errors,'provider_nodes':{f'{op}:{provider}':n for (op,provider),n in placement.items()}}
            print('PROBE',device,mode,name,max(e['relative_l2'] for e in errors),flush=True)
            del session
        result.append(record)
    (ROOT/'results'/'runtime'/'probe.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
