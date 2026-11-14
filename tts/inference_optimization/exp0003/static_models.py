"""D: 固定activation scaleを使うINT8/SmoothQuant。"""
import hashlib
import random
import time
from pathlib import Path


def name_for(device, alpha):
    return f"{device}_static_int8" if alpha is None else f"{device}_smoothquant_{str(alpha).replace('.', 'p')}"


def prepare(torch, pipe, t2s, vocab, seed, alpha, root):
    import torchao
    from torchao.quantization import quantize_, Int8StaticActivationInt8WeightConfig
    from torchao.quantization.granularity import PerTensor, PerRow
    from torchao.prototype.smoothquant import SmoothQuantConfig
    from benchmark import CHECKPOINTS, TTS_DIR
    from lib_common.data import list_jsut, split
    from exp0002.common.t2s_masked import mask_ratio
    device = pipe.device
    condition = name_for(device, alpha)
    t2s.requires_grad_(False)
    pipe.s2m.requires_grad_(False)
    hashes = {}
    for key, path in {**CHECKPOINTS, **{n:TTS_DIR/'lib_common'/f'{n}.py' for n in ['audio','text','vocoder']}}.items():
        with path.open('rb') as handle:
            hashes[key] = hashlib.file_digest(handle,'sha256').hexdigest()
    signature = {'schema':1,'source_sha256':hashes,'torch':str(torch.__version__),
                 'torchao':torchao.__version__,'device':device,'alpha':alpha}
    cache = root/'runs'/'static'/f'{condition}.pt'
    names={label:[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and m.out_features % 8 == 0 and not(label=='t2s' and n=='head')]
           for label,model in [('t2s',t2s),('s2m',pipe.s2m)]}
    if cache.exists():
        state = torch.load(cache,map_location=device,weights_only=False)
        if state['signature'] != signature:
            raise RuntimeError('D checkpointの実験条件が不一致')
        if state['metadata']['layers'] != names:
            raise RuntimeError('D checkpointの量子化対象層が不一致')
        t2s.load_state_dict(state['t2s'],assign=True)
        pipe.s2m.load_state_dict(state['s2m'],assign=True)
        return state['metadata']
    start=time.perf_counter()
    train=sorted(split(list_jsut(),'train'),key=lambda u:(len(u.text),u.id))
    rng=random.Random(0)
    selected=[train[rng.randrange(i*len(train)//8,(i+1)*len(train)//8)] for i in range(8)]
    reference_cache=root/'runs'/'static'/f'reference_{device}.pt'
    reference_signature={**signature,'alpha':None,'ids':[u.id for u in selected]}
    references=[]
    if reference_cache.exists():
        state=torch.load(reference_cache,map_location=device,weights_only=False)
        if state['signature'] != reference_signature:
            raise RuntimeError('D calibration入力の条件が不一致')
        references=state['references']
    else:
        seed(0)
        with torch.no_grad():
            for u in selected:
                ids=[torch.tensor(vocab.encode(u.text))]
                durations=t2s.predict_durations(ids)
                codes=t2s.generate(ids,durations=durations)
                mel=pipe.codes_to_mel(codes,batch=1)[0]
                references.append((ids,durations,codes,pipe.normalizer.normalize(mel)))
        reference_cache.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'signature':reference_signature,'references':references},reference_cache)
    models={'t2s':t2s,'s2m':pipe.s2m}
    base=Int8StaticActivationInt8WeightConfig(granularity=[PerTensor(),PerRow()],set_inductor_config=False)
    kwargs=base.get_act_quant_kwargs()
    base.get_act_quant_kwargs=lambda:{**vars(kwargs),'is_symmetric':True}
    ranks={}
    hooks=[]
    def rank_hook(key):
        def record(module,inputs):
            rank=inputs[0].ndim
            if key in ranks and ranks[key]!=rank:
                raise RuntimeError(f'入力rankが不定: {key}')
            ranks[key]=rank
        return record
    def transform(step):
        for label,model in models.items():
            quantize_(model,SmoothQuantConfig(base,step=step,alpha=alpha,use_running_absmax=True),
                      filter_fn=lambda m,n,label=label:n in names[label])
    transform('prepare')
    for label,model in models.items():
        for n in names[label]:
            hooks.append(model.get_submodule(n).register_forward_pre_hook(rank_hook(f'{label}.{n}')))
    def observe():
        seed(0)
        with torch.no_grad():
            for ids,durations,codes,mel in references:
                ids=[x.cpu() for x in ids]
                t2s.predict_durations(ids)
                h,_=t2s.encode(ids)
                aligned=t2s.align(h,durations)
                for value in [1.,.5,.04]:
                    c=codes[0].cpu()
                    mask=torch.rand(c.shape)<mask_ratio(torch.tensor(value))
                    xt=[torch.where(mask,t2s.n_codes,c)]
                    for drop in [False,True]:
                        t2s.hidden(ids,xt,torch.tensor([value]),torch.tensor([drop]),aligned)
                audio_ids=codes[0].unsqueeze(0).to(device)
                cond=pipe.s2m.regulator(pipe.codec.decode_hidden(audio_ids),len(mel))
                target=mel.unsqueeze(0).to(device)
                noise=torch.randn_like(target)
                pad=torch.zeros(2,len(mel),dtype=torch.bool,device=device)
                for value in [0.,.5,.96]:
                    xt=(1-value)*noise+value*target
                    pipe.s2m.velocity(torch.cat([xt,xt]),torch.cat([cond,torch.zeros_like(cond)]),
                                      torch.full((2,),value,device=device),pad)
    print(f'{condition}: calibration pass 1',flush=True)
    observe()
    transform('prepare_for_smoothquant_smoothing_factor')
    print(f'{condition}: calibration pass 2',flush=True)
    observe()
    for hook in hooks:
        hook.remove()
    transform('convert')
    scales={}
    for label,model in models.items():
        for n in names[label]:
            w=model.get_submodule(n).weight
            shape=[1]*ranks[f'{label}.{n}']
            w.act_quant_scale=w.act_quant_scale.reshape(shape)
            if w.act_quant_zero_point is not None:
                w.act_quant_zero_point=w.act_quant_zero_point.reshape(shape)
            if alpha is None:
                w.act_pre_scale=None
            scales[f'{label}.{n}']=w.act_quant_scale.item()
    metadata={'signature':signature,'calibration_ids':[u.id for u in selected], 'seed':0,
              'texts':8,'passes':2,'weight_granularity':'per-row','activation_granularity':'per-tensor',
              'mapping':'symmetric','excluded_t2s':['head'],'layers':names,'activation_scales':scales,
              'calibration_s':time.perf_counter()-start}
    cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'signature':signature,'metadata':metadata,'t2s':t2s.state_dict(),'s2m':pipe.s2m.state_dict()},cache)
    state=torch.load(cache,map_location=device,weights_only=False)
    first=names['t2s'][0]
    module=t2s.get_submodule(first)
    x=torch.randn(*([1]*(ranks[f't2s.{first}']-1)),module.in_features,device=device)
    with torch.no_grad():
        expected=module(x)
    t2s.load_state_dict(state['t2s'],assign=True)
    pipe.s2m.load_state_dict(state['s2m'],assign=True)
    with torch.no_grad():
        if not torch.equal(expected,t2s.get_submodule(first)(x)):
            raise RuntimeError('D checkpoint往復検査失敗')
    print(f'{condition}: calibration/checkpoint OK',flush=True)
    return metadata
