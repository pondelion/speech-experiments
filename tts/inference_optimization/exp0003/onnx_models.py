"""G: 可変長の計算部分をONNXへ移し、生成ループは共通に保つ。"""
from pathlib import Path
import gc
import json
import types

ROOT=Path(__file__).resolve().parent
MODELS=ROOT/'runs'/'onnx'


def export_models(torch,pipe,t2s):
    import onnx
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=t2s.encoder
        def forward(self,x,mask):
            return self.encoder(inputs_embeds=x,attention_mask=mask).last_hidden_state
    class Blocks(torch.nn.Module):
        def __init__(self):
            super().__init__();self.blocks=t2s.blocks
        def forward(self,x,t,pad):
            for block in self.blocks:x=block(x,t,pad)
            return x
    class Velocity(torch.nn.Module):
        def __init__(self):
            super().__init__();self.model=pipe.s2m
        def forward(self,x,cond,t,pad):
            return self.model.velocity(x,cond,t,pad)
    assert pipe.s2m.text_dim==0
    specs=[('encoder',Encoder(),(torch.randn(2,32,512),torch.ones(2,32,dtype=torch.int64)),['x','mask'],{'x':{0:'batch',1:'length'},'mask':{0:'batch',1:'length'}}),
           ('blocks',Blocks(),(torch.randn(2,64,512),torch.randn(2,512),torch.zeros(2,64,dtype=torch.bool)),['x','t','pad'],{'x':{0:'batch',1:'length'},'t':{0:'batch'},'pad':{0:'batch',1:'length'}}),
           ('velocity',Velocity(),(torch.randn(2,64,80),torch.randn(2,64,pipe.s2m.dim),torch.rand(2),torch.zeros(2,64,dtype=torch.bool)),['x','cond','t','pad'],{'x':{0:'batch',1:'length'},'cond':{0:'batch',1:'length'},'t':{0:'batch'},'pad':{0:'batch',1:'length'}})]
    MODELS.mkdir(parents=True,exist_ok=True)
    result={}
    for name,model,inputs,names,axes in specs:
        path=MODELS/f'{name}_fp32.onnx'
        if not path.exists():
            with torch.inference_mode():
                torch.onnx.export(model.eval(),inputs,str(path),input_names=names,output_names=['y'],
                    dynamic_axes={**axes,'y':{0:'batch',1:'length'}},opset_version=18,dynamo=False,
                    external_data=False)
        graph=onnx.load(str(path))
        # Normalize Gemm into MatMul + Add so rank-2 Linear weights are eligible
        # for the same quantizer as rank-3 Linear weights.
        constants={x.name:x for x in graph.graph.initializer};nodes=[];changed=False
        for node in graph.graph.node:
            if node.op_type!='Gemm':nodes.append(node);continue
            attr={a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}
            assert attr.get('alpha',1)==attr.get('beta',1)==1 and not attr.get('transA',0)
            weight=onnx.numpy_helper.to_array(constants[node.input[1]])
            if attr.get('transB',0):weight=weight.T
            key=node.input[1]+'_matmul'
            graph.graph.initializer.append(onnx.numpy_helper.from_array(weight.copy(),key))
            output=node.output[0]+'_linear' if len(node.input)==3 else node.output[0]
            nodes.append(onnx.helper.make_node('MatMul',[node.input[0],key],[output],name=node.name+'_matmul'))
            if len(node.input)==3:nodes.append(onnx.helper.make_node('Add',[output,node.input[2]],list(node.output),name=node.name+'_bias'))
            changed=True
        if changed:
            del graph.graph.node[:];graph.graph.node.extend(nodes)
            used={i for n in nodes for i in n.input}
            kept=[x for x in graph.graph.initializer if x.name in used]
            del graph.graph.initializer[:];graph.graph.initializer.extend(kept)
            onnx.save(graph,str(path))
            for mode in ['int8','int4']:(MODELS/f'{name}_{mode}.onnx').unlink(missing_ok=True)
        onnx.checker.check_model(graph)
        result[name]={'bytes':path.stat().st_size,'nodes':len(graph.graph.node)}
        print('EXPORTED',name,result[name],flush=True)
    (ROOT/'results'/'runtime').mkdir(parents=True,exist_ok=True)
    (ROOT/'results'/'runtime'/'export.json').write_text(json.dumps(result,indent=2)+'\n')


def quantized_paths(mode):
    import onnx
    from onnxruntime.quantization import quantize_dynamic,QuantType
    files=[]
    for name in ['encoder','blocks','velocity']:
        src=MODELS/f'{name}_fp32.onnx';dst=MODELS/f'{name}_{mode}.onnx'
        if mode=='fp32':dst=src
        elif not dst.exists():
            if mode=='int8':
                quantize_dynamic(str(src),str(dst),per_channel=True,weight_type=QuantType.QInt8,
                                 op_types_to_quantize=['MatMul'],extra_options={'MatMulConstBOnly':True})
            elif mode=='int4':
                from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
                quantizer=MatMulNBitsQuantizer(str(src),block_size=128,is_symmetric=False,accuracy_level=4,
                                             op_types_to_quantize=('MatMul',))
                quantizer.process();quantizer.model.save_model_to_file(str(dst),use_external_data_format=False)
        onnx.checker.check_model(onnx.load(str(dst)))
        files.append(dst)
    return files


def prepare(torch,pipe,t2s,device,mode):
    if mode=='torch':return {'backend':'pytorch_eager'},t2s
    import numpy as np
    import onnxruntime as ort
    from transformers.modeling_outputs import BaseModelOutput
    ort.preload_dlls()
    paths=quantized_paths(mode)
    sessions={};metadata={'backend':'onnxruntime','version':ort.__version__,'precision':mode,'graphs':{}}
    for name,path in zip(['encoder','blocks','velocity'],paths):
        options=ort.SessionOptions();options.intra_op_num_threads=8;options.inter_op_num_threads=1
        providers=['CPUExecutionProvider']
        if device=='cuda':
            providers=[('CUDAExecutionProvider',{'device_id':0,'use_tf32':0,
                'user_compute_stream':str(torch.cuda.current_stream().cuda_stream)}),'CPUExecutionProvider']
        session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        session.disable_fallback()
        if device=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():
            raise RuntimeError('CUDA provider was not activated')
        sessions[name]=session
        metadata['graphs'][name]={'bytes':path.stat().st_size,'providers':session.get_providers()}
    class Core(torch.nn.Module):
        def __init__(self,name):super().__init__();self.name=name
        def run(self,values,shape):
            session=sessions[self.name];binding=session.io_binding()
            contiguous=[x.contiguous() for x in values]
            dtypes={torch.float32:np.float32,torch.int64:np.int64,torch.bool:np.bool_}
            for spec,x in zip(session.get_inputs(),contiguous):
                binding.bind_input(spec.name,device,0,dtypes[x.dtype],tuple(x.shape),x.data_ptr())
            output=torch.empty(shape,device=device,dtype=torch.float32)
            binding.bind_output('y',device,0,np.float32,tuple(output.shape),output.data_ptr())
            session.run_with_iobinding(binding)
            return output
        def forward(self,x,t,pad):return self.run([x,t,pad],x.shape)
    class ORTEncoder(Core):
        def forward(self,inputs_embeds,attention_mask):
            return BaseModelOutput(last_hidden_state=self.run([inputs_embeds,attention_mask],inputs_embeds.shape))
    t2s.encoder=ORTEncoder('encoder')
    t2s.blocks=torch.nn.ModuleList([Core('blocks')])
    velocity=Core('velocity')
    def forward_velocity(self,x,cond,t,pad,text=None,text_pad=None):
        return velocity.run([x,cond,t,pad],x.shape)
    pipe.s2m.velocity=types.MethodType(forward_velocity,pipe.s2m)
    for name in ['in_proj','t_mlp','blocks','norm_out','out']:
        delattr(pipe.s2m,name)
    # Keep session objects alive without retaining a second copy of torch weights.
    t2s._ort_sessions=sessions
    gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return metadata,t2s
