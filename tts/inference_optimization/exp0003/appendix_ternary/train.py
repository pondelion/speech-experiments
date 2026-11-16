"""Fの固定予算比較。既存トークンと音素ラベルだけを使い、音声は読まない。"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'exp0001'))
from benchmark import build_models,CHECKPOINTS
from appendix_ternary.models import qat_prepare


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['ft','qat','kd'],required=True)
    parser.add_argument('--steps',type=int,default=1000)
    args=parser.parse_args()
    out=ROOT/'runs'/'ternary';out.mkdir(parents=True,exist_ok=True)
    target=out/f'train_{args.mode}.pt'
    if target.exists():
        print(f'SKIP training {args.mode}',flush=True);return
    torch,pipe,model,vocab,seed=build_models('cuda')
    torch.set_num_threads(8)
    from lib_common.data import list_jsut,split
    from lib_common.utils import length_batches
    from exp0002.common.utils import prepare,TOKEN_SEC
    cache=CHECKPOINTS['codec'].parent/'tokens.pt'
    tokens=torch.load(cache,map_location='cpu',weights_only=False)
    data={}; ids={}
    for group,count in [('train',512),('val',64)]:
        utterances=sorted([u for u in split(list_jsut(),group) if u.id in tokens],key=lambda u:(len(tokens[u.id]),u.id))
        rng=random.Random(0)
        selected=[utterances[rng.randrange(i*len(utterances)//count,(i+1)*len(utterances)//count)] for i in range(count)]
        d=prepare(selected,vocab,None,None,'cuda',True,[tokens[u.id].long() for u in selected],TOKEN_SEC)
        data[group]=d;ids[group]=[selected[i].id for i in d['keep']]
    assert not set(ids['train'])&set(ids['val'])
    source={}
    for name,path in {**CHECKPOINTS,'tokens':cache}.items():
        with path.open('rb') as f:source[name]=hashlib.file_digest(f,'sha256').hexdigest()
    del pipe,tokens
    torch.cuda.empty_cache()
    seed(0)
    import copy,types
    teacher=copy.deepcopy(model).eval().requires_grad_(False) if args.mode=='kd' else None
    names=qat_prepare(model) if args.mode in ['qat','kd'] else []
    current_durations=[]
    if teacher is not None:
        original_hidden=model.hidden
        def hidden(self,text_ids,xt,t,drop_text,audio_cond=None):
            value,valid=original_hidden(text_ids,xt,t,drop_text,audio_cond)
            if self.training:
                with torch.no_grad():
                    th,_=teacher.encode(text_ids)
                    th,_=teacher.hidden(text_ids,xt,t,drop_text,teacher.align(th,current_durations))
                    selected=torch.nn.utils.rnn.pad_sequence([x==self.n_codes for x in xt],batch_first=True).to(self.device)
                    target=teacher.head(th[selected]).float()
                student=self.head(value[selected]).float()
                self._kd_loss=torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(student/2,dim=-1),
                    torch.nn.functional.softmax(target/2,dim=-1),reduction='batchmean')*4
            return value,valid
        model.hidden=types.MethodType(hidden,model)
    seed(0)  # QAT module construction consumes RNG; align training masks/dropout.
    model.train();model.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    tr=data['train']; lengths=[len(t)+len(c) for t,c in zip(tr['texts'],tr['codes'])]
    def loss_for(d,b):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            current_durations[:]=[d['durations'][i] for i in b]
            base=model([d['texts'][i] for i in b],[d['codes'][i] for i in b],current_durations)[0]
            return base+model._kd_loss if teacher is not None and model.training else base
    def validate():
        model.eval()
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            seed(123)
            d=data['val'];lens=[len(t)+len(c) for t,c in zip(d['texts'],d['codes'])]
            total=0.;n=0
            for b in length_batches(lens,4096,shuffle=False):
                total+=float(loss_for(d,b))*len(b);n+=len(b)
        model.train();return total/n
    log=[{'step':0,'val_loss':validate()}]
    start=time.perf_counter();step=0;epoch=0
    torch.cuda.reset_peak_memory_stats()
    while step<args.steps:
        for batch in length_batches(lengths,4096,shuffle=True,seed=epoch):
            step+=1
            optimizer.param_groups[0]['lr']=3e-5*min(step/50,1)*max(.1,(args.steps-step)/max(1,args.steps-50))
            optimizer.zero_grad(set_to_none=True)
            loss=loss_for(tr,batch)
            if not bool(torch.isfinite(loss)):raise RuntimeError('nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)):raise RuntimeError('nonfinite gradient')
            optimizer.step()
            if step%200==0 or step==args.steps:
                row={'step':step,'train_loss':float(loss.detach()),'val_loss':validate()}
                log.append(row);print(args.mode,row,flush=True)
            if step==args.steps:break
        epoch+=1
    torch.cuda.synchronize()
    metadata={'mode':args.mode,'steps':args.steps,'seed':0,'learning_rate':3e-5,'optimizer':'AdamW',
        'weight_decay':.01,'grad_clip':1.,'warmup_steps':50,'max_tokens':4096,'checkpoint_selection':'fixed final step',
        'training_ids':ids['train'],'validation_ids':ids['val'],'source_sha256':source,
        'quantized_layers':names,'training_s_including_validation':time.perf_counter()-start,
        'peak_vram_bytes':torch.cuda.max_memory_allocated(),'master_weights':'float32','autocast':'bfloat16','log':log,'distillation':{'type':'masked audio-token logits KL','temperature':2,'weight':1,'teacher':'frozen original T2S'} if teacher is not None else None,
        'torch':str(torch.__version__)}
    torch.save({'model':model.state_dict(),'metadata':metadata},target)
    result=ROOT/'results'/'ternary';result.mkdir(parents=True,exist_ok=True)
    (result/f'train_{args.mode}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'DONE training {args.mode}',flush=True)


if __name__=='__main__':main()
