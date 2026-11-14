"""Dの速度・CER・メモリを同一条件で測る。"""
import argparse
import gc
import io
import json
from pathlib import Path
import resource
import statistics
import sys
import time
import types

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'exp0001'))
from benchmark import build_models, one_run, load_inputs, summarize, environment
from static_models import prepare, name_for


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--mode',choices=['fp32','static','sq05','sq075'],required=True)
    parser.add_argument('--phase',choices=['bench','quality','memory'],required=True)
    parser.add_argument('--eager',action='store_true')
    args=parser.parse_args()
    alpha={'static':None,'sq05':.5,'sq075':.75}.get(args.mode)
    condition=f'{args.device}_d_fp32' if args.mode=='fp32' else name_for(args.device,alpha)
    if not args.eager:
        condition+='_clinear'
    out=ROOT/'results'/'static'
    out.mkdir(parents=True,exist_ok=True)
    output=out/f'{args.phase}_{condition}.json'
    if output.exists():
        print(f'SKIP {output.name}',flush=True)
        return
    torch,pipe,t2s,vocab,seed=build_models(args.device)
    torch.set_num_threads(8)
    meta={} if args.mode=='fp32' else prepare(torch,pipe,t2s,vocab,seed,alpha,ROOT)
    if not args.eager:
        torch._dynamo.config.recompile_limit=64
        compiled=torch.compile(torch.nn.functional.linear,fullgraph=True,dynamic=True)
        def forward(self,x):
            # CUDA INT8 GEMM requires more than 16 rows. Pad only tiny inputs;
            # discard added rows before attention so model semantics are unchanged.
            if x.is_cuda and x.numel() // x.shape[-1] <= 16:
                rows=x.shape[-2]
                padded=torch.nn.functional.pad(x,(0,0,0,32-rows))
                return compiled(padded,self.weight,self.bias)[..., :rows, :]
            return compiled(x,self.weight,self.bias)
        for model in [t2s,pipe.s2m]:
            for layer in model.modules():
                if isinstance(layer,torch.nn.Linear):
                    layer.forward=types.MethodType(forward,layer)
    result={'condition':condition,'phase':args.phase,'environment':environment(torch,args),'quantization':meta,
            'linear_backend':'eager' if args.eager else 'torch.compile(fullgraph=True,dynamic=True)'}
    if args.phase=='bench':
        sizes={}
        for label,model in [('t2s',t2s),('s2m',pipe.s2m)]:
            buffer=io.BytesIO()
            torch.save(model.state_dict(),buffer)
            sizes[label]=buffer.tell()
        del buffer
        result.update(state_bytes=sizes,warmup=1,repeats=3,cases={})
        samples=[]
        for item in load_inputs('all'):
            warmup_start=time.perf_counter()
            one_run(torch,pipe,t2s,vocab,seed,item['text'],args.device,0,'float32')
            warmup_s=time.perf_counter()-warmup_start
            times=[]; shapes=[]
            for i in range(3):
                path=ROOT/'results'/'samples'/'eval_texts'/condition/f"{item['id']}.mp3"
                stages,shape=one_run(torch,pipe,t2s,vocab,seed,item['text'],args.device,i+1,'float32',path if i==2 else None)
                times.append(stages); shapes.append(shape)
            result['cases'][item['id']]={'stages':{k:summarize([v[k] for v in times]) for k in times[0]},
                'warmup_including_compile_s':warmup_s,
                'rtf':statistics.median(v['rtf'] for v in shapes),
                'finite':all(v['mel_all_finite'] and v['wave_all_finite'] for v in shapes)}
            print(f"{condition} bench {item['id']}: {result['cases'][item['id']]['stages']['end_to_end_s']['median_s']:.3f}s",flush=True)
            samples.append({'group':'eval_texts','condition':condition,'id':item['id'],'text':item['text'],
                'category':item['id'],'file':str(path.relative_to(ROOT/'results'/'samples'))})
        index=ROOT/'results'/'samples'/'index.json'
        rows=json.loads(index.read_text())
        index.write_text(json.dumps([r for r in rows if r['condition']!=condition]+samples,indent=2,ensure_ascii=False)+'\n')
    elif args.phase=='memory':
        gc.collect()
        one_run(torch,pipe,t2s,vocab,seed,load_inputs('long')[0]['text'],args.device,0,'float32')
        if args.device=='cuda':
            torch.cuda.synchronize()
            result['load_convert_warmup_peak_vram_bytes']=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        one_run(torch,pipe,t2s,vocab,seed,load_inputs('long')[0]['text'],args.device,0,'float32')
        result['process_peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if args.device=='cuda':
            result['inference_peak_vram_bytes']=torch.cuda.max_memory_allocated()
    else:
        import soundfile as sf
        from lib_common.data import list_jsut,split
        from lib_common.eval_sets import load_eval_texts
        from lib_common.audio import SR
        from lib_common.asr_eval import ASR,cer_scores
        groups={'test':split(list_jsut(),'test'),'eval_texts':load_eval_texts(['practical'])}
        audio={}; result.update(seed=0,t2s_batch=16,s2m_batch=4,groups={})
        for group,items in groups.items():
            seed(0); codes=[]
            with torch.inference_mode():
                for start in range(0,len(items),16):
                    ids=[torch.tensor(vocab.encode(u.text)) for u in items[start:start+16]]
                    durations=t2s.predict_durations(ids)
                    if any(bool((d<0).any()) or int(d.sum())>4096 for d in durations):
                        raise RuntimeError('生成長の安全上限4096トークンを超過')
                    codes+=t2s.generate(ids,durations=durations)
                seed(0)
                mels=pipe.codes_to_mel(codes,batch=4,n_steps=25,cfg=.7)
            paths=[]; finite=True
            for item,mel in zip(items,mels):
                wave=pipe.mel_to_wave(mel.float())
                finite=finite and bool(torch.isfinite(mel).all() and torch.isfinite(wave).all())
                path=ROOT/'runs'/'static_audio'/condition/group/f'{item.id}.wav'
                path.parent.mkdir(parents=True,exist_ok=True)
                sf.write(path,wave.numpy(),SR); paths.append(str(path))
            audio[group]=paths
            result['groups'][group]={'utterances':len(items),'finite':finite}
            print(f'{condition} generated {group}',flush=True)
        asr=ASR('cuda')
        for group,paths in audio.items():
            scores=cer_scores([u.text for u in groups[group]],asr.transcribe(paths,batch_size=16))
            result['groups'][group]['cer']=scores
            print(f"{condition} CER {group}: {scores['reading']:.6f}",flush=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'DONE {args.phase} {condition}',flush=True)


if __name__=='__main__':
    main()
