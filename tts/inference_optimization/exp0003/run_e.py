"""E: 部品単位・入出力側の層を高精度に残す比較。"""
import argparse
import fcntl
import gc
import io
import json
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'exp0001'))
from benchmark import build_models, one_run, load_inputs, summarize, environment
from mixed_models import prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    parser.add_argument('--mode',choices=['baseline','all','t2s','s2m','boundary'],required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--phase',choices=['bench','quality','memory'],required=True)
    args=parser.parse_args()
    condition=f'{args.device}_e_{args.mode}'
    dtype='bfloat16' if args.device=='cuda' else 'float32'
    out=ROOT/'results'/'mixed'
    out.mkdir(parents=True,exist_ok=True)
    output=out/f'{args.phase}_{condition}.json'
    lock_dir=ROOT/'runs'/'mixed_locks'
    lock_dir.mkdir(parents=True,exist_ok=True)
    lock=(lock_dir/f'{args.phase}_{condition}.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if output.exists():
        print(f'SKIP {output.name}',flush=True)
        return
    torch,pipe,t2s,vocab,seed=build_models(args.device)
    torch.set_num_threads(8)
    meta,t2s=prepare(torch,pipe,t2s,args.device,args.mode)
    result={'condition':condition,'phase':args.phase,'environment':environment(torch,args),
            'quantization':meta,'dtype':dtype,'linear_backend':'eager'}
    if args.phase=='bench':
        sizes={}
        for label,model in [('t2s',t2s),('s2m',pipe.s2m)]:
            buffer=io.BytesIO()
            torch.save(model.state_dict(),buffer)
            sizes[label]=buffer.tell()
        del buffer
        result.update(state_bytes=sizes,warmup=1,repeats=args.repeats,cases={})
        samples=[]
        for item in load_inputs('all'):
            warmup_start=time.perf_counter()
            one_run(torch,pipe,t2s,vocab,seed,item['text'],args.device,0,dtype)
            warmup_s=time.perf_counter()-warmup_start
            times=[]; shapes=[]
            for i in range(args.repeats):
                path=ROOT/'results'/'samples'/'eval_texts'/condition/f"{item['id']}.mp3"
                stages,shape=one_run(torch,pipe,t2s,vocab,seed,item['text'],args.device,i+1,dtype,path if i==args.repeats-1 else None)
                times.append(stages); shapes.append(shape)
            result['cases'][item['id']]={'stages':{k:summarize([v[k] for v in times]) for k in times[0]},
                'warmup_s':warmup_s,
                'rtf':statistics.median(v['rtf'] for v in shapes),
                'finite':all(v['mel_all_finite'] and v['wave_all_finite'] for v in shapes)}
            output.with_suffix('.partial.json').write_text(json.dumps(result,indent=2)+'\n')
            print(f"{condition} bench {item['id']}: {result['cases'][item['id']]['stages']['end_to_end_s']['median_s']:.3f}s",flush=True)
            samples.append({'group':'eval_texts','condition':condition,'id':item['id'],'text':item['text'],
                'category':item['id'],'file':str(path.relative_to(ROOT/'results'/'samples'))})
        index=ROOT/'results'/'samples'/'index.json'
        rows=json.loads(index.read_text())
        index.write_text(json.dumps([r for r in rows if r['condition']!=condition]+samples,indent=2,ensure_ascii=False)+'\n')
    elif args.phase=='memory':
        gc.collect()
        one_run(torch,pipe,t2s,vocab,seed,load_inputs('long')[0]['text'],args.device,0,dtype)
        if args.device=='cuda':
            torch.cuda.synchronize()
            result['load_convert_warmup_peak_vram_bytes']=torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        one_run(torch,pipe,t2s,vocab,seed,load_inputs('long')[0]['text'],args.device,0,dtype)
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
            with torch.inference_mode(), (torch.autocast('cuda', dtype=torch.bfloat16) if args.device=='cuda' else __import__('contextlib').nullcontext()):
                for start in range(0,len(items),16):
                    ids=[torch.tensor(vocab.encode(u.text)) for u in items[start:start+16]]
                    durations=t2s.predict_durations(ids)
                    if any(bool((d<0).any()) or int(d.sum())>4096 for d in durations):
                        raise RuntimeError('生成長の安全上限4096トークンを超過')
                    codes+=t2s.generate(ids,durations=durations)
                    print(f'{condition} T2S {group} {min(start+16,len(items))}/{len(items)}',flush=True)
                seed(0)
                mels=pipe.codes_to_mel(codes,batch=4,n_steps=25,cfg=.7)
            paths=[]; finite=True
            for item,mel in zip(items,mels):
                wave=pipe.mel_to_wave(mel.float())
                finite=finite and bool(torch.isfinite(mel).all() and torch.isfinite(wave).all())
                path=ROOT/'runs'/'mixed_audio'/condition/group/f'{item.id}.wav'
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
    output.with_suffix('.partial.json').unlink(missing_ok=True)
    print(f'DONE E {args.phase} {condition}',flush=True)


if __name__=='__main__':
    main()
