"""既存torchaoのAWQ/GPTQを、学習splitのテキストでcalibrationする。"""
from contextlib import nullcontext, redirect_stdout, contextmanager
import hashlib
import io
import random
import time
import types


@contextmanager
def fp32_hessian(torch, enabled):
    if not enabled:
        yield
        return
    from torchao.prototype.gptq.observer import GPTQObserverTensor
    original = GPTQObserverTensor._update_single_hessian
    def update(x, hessian, batches):
        with torch.autocast(device_type="cuda", enabled=False):
            return original(x.float(), hessian, batches)
    GPTQObserverTensor._update_single_hessian = staticmethod(update)
    try:
        yield
    finally:
        GPTQObserverTensor._update_single_hessian = staticmethod(original)


def calibrated_models(torch, pipe, t2s, vocab, seed, method, bits, group_size, root, condition):
    import torchao
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    from torchao.prototype.awq import AWQConfig
    from torchao.prototype.gptq import GPTQConfig
    from benchmark import CHECKPOINTS
    from lib_common.data import list_jsut, split
    from exp0002.common.t2s_masked import mask_ratio

    dtype = next(t2s.parameters()).dtype
    t2s.requires_grad_(False)
    pipe.s2m.requires_grad_(False)
    device = pipe.device
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if dtype == torch.bfloat16 else nullcontext()

    train = sorted(split(list_jsut(), "train"), key=lambda u: (len(u.text), u.id))
    rng = random.Random(0)
    selected = [train[rng.randrange(i*len(train)//8, (i+1)*len(train)//8)] for i in range(8)]
    hashes = {}
    for name, path in CHECKPOINTS.items():
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    signature = {"schema": 1, "method": method, "bits": bits, "group_size": group_size,
                 "dtype": str(dtype), "torch": str(torch.__version__), "torchao": torchao.__version__,
                 "checkpoint_sha256": hashes, "calibration_ids": [u.id for u in selected]}
    cache = root / "runs" / "calibration" / f"{condition}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise RuntimeError("calibration cacheの条件が不一致。別condition名で再生成してください")
        t2s.load_state_dict(saved["t2s"], assign=True)
        pipe.s2m.load_state_dict(saved["s2m"], assign=True)
        return saved["metadata"]

    start = time.perf_counter()
    references = []
    seed(0)
    with torch.inference_mode(), amp():
        for u in selected:
            ids = [torch.tensor(vocab.encode(u.text))]
            durations = t2s.predict_durations(ids)
            codes = t2s.generate(ids, durations=durations)
            mel = pipe.codes_to_mel(codes, n_steps=25, cfg=0.7, batch=1)[0]
            references.append((ids, durations, codes, pipe.normalizer.normalize(mel)))
    reference_s = time.perf_counter() - start

    base = Int8WeightOnlyConfig(set_inductor_config=False) if bits == 8 else Int4WeightOnlyConfig(group_size=group_size, set_inductor_config=False)
    if bits == 4 and method != "gptq":
        base.int4_packing_format = type(base.int4_packing_format).TILE_PACKED_TO_4D
    models = {"t2s": t2s, "s2m": pipe.s2m}
    meta = {"signature": signature, "calibration_split": "train", "texts": 8,
            "reference_generation_s": reference_s, "states_t2s": [1.0, 0.5, 0.04],
            "states_s2m": [0.0, 0.5, 0.96], "activation_rows_per_layer_cap": 256 if method == "awq" else None,
            "observation": "all layers observe unquantized upstream outputs; not sequential GPTQ",
            "scale_search_space_size": 20 if method == "awq" else None,
            "percdamp": 0.01 if method == "gptq" else None, "layers": {}}

    # AWQObserverの保持するactivationを均等サンプルで制限する。探索式自体はlibrary実装を使う。
    def bounded_observe(self, inp, output):
        flat = inp.detach().reshape(-1, inp.shape[-1])
        remaining = 256 - getattr(self, "sampled_rows", 0)
        count = min(4, remaining, flat.shape[0])
        if count:
            indices = torch.linspace(0, flat.shape[0]-1, count, device=flat.device).long()
            # observerはautocast前の入力を見るため、実際のLinear演算dtypeへそろえる。
            self.inputs.append(flat[indices].to(dtype=self.weight.dtype).unsqueeze(0).cpu())
            self.sampled_rows = getattr(self, "sampled_rows", 0) + count

    for label, model in models.items():
        names, excluded = [], []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if (label == "t2s" and name == "head") or (bits == 4 and (module.in_features % group_size or module.out_features % 8)):
                    excluded.append(name)
                else:
                    names.append(name)
        meta["layers"][label] = {"quantized_layers": names, "excluded_layers": excluded}
        cfg = AWQConfig(base, step="prepare") if method == "awq" else GPTQConfig(step="observe", base_config=base)
        quantize_(model, cfg, filter_fn=lambda module, name, names=names: name in names)
        if method == "awq":
            for name in names:
                observer = model.get_submodule(name).act_obs
                observer.forward = types.MethodType(bounded_observe, observer)

    print(f"calibration {condition}: observing 8 training texts", flush=True)
    seed(0)
    with torch.inference_mode(), amp(), fp32_hessian(torch, method == "gptq"):
        for ids, durations, codes, mel in references:
            # 正確な推論経路のencoder/length regulatorを観測する。
            t2s.predict_durations(ids)
            h, _ = t2s.encode(ids)
            aligned = t2s.align(h, durations)
            for value in [1.0, 0.5, 0.04]:
                mask = torch.rand(codes[0].shape) < mask_ratio(torch.tensor(value))
                xt = [torch.where(mask, t2s.n_codes, codes[0])]
                for drop in [False, True]:
                    t2s.hidden(ids, xt, torch.tensor([value]), torch.tensor([drop]), aligned)
            ids_audio = codes[0].unsqueeze(0).to(device)
            cond = pipe.s2m.regulator(pipe.codec.decode_hidden(ids_audio), len(mel))
            target = mel.unsqueeze(0).to(device)
            noise = torch.randn_like(target)
            pad = torch.zeros(2, len(mel), dtype=torch.bool, device=device)
            for value in [0.0, 0.5, 0.96]:
                xt = (1-value)*noise + value*target
                pipe.s2m.velocity(torch.cat([xt, xt]), torch.cat([cond, torch.zeros_like(cond)]),
                                  torch.full((2,), value, device=device), pad)

    for label, model in models.items():
        names = meta["layers"][label]["quantized_layers"]
        if method == "awq":
            meta["layers"][label]["observed_rows"] = {
                name: model.get_submodule(name).act_obs.sampled_rows for name in names}
        print(f"calibration {condition}: converting {label}", flush=True)
        cfg = AWQConfig(base, step="convert") if method == "awq" else GPTQConfig(step="convert", base_config=base)
        with torch.no_grad(), redirect_stdout(io.StringIO()):
            quantize_(model, cfg, filter_fn=lambda module, name, names=names: name in names)
        if method == "gptq" and bits == 4:
            from repack import repack_model
            meta["layers"][label]["repack_checks"] = repack_model(torch, model, names)
    torch.cuda.synchronize()
    meta["total_calibration_s"] = time.perf_counter() - start
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"signature": signature, "metadata": meta, "t2s": t2s.state_dict(),
                "s2m": pipe.s2m.state_dict()}, cache)
    # 同一Tensor subclassと付加scaleがcheckpoint往復で保存されることを確認する。
    saved = torch.load(cache, map_location=device, weights_only=False)
    first = meta["layers"]["t2s"]["quantized_layers"][0]
    layer = t2s.get_submodule(first)
    x = torch.randn(1, layer.in_features, device=device, dtype=dtype)
    with torch.inference_mode(), amp():
        expected = layer(x).clone()
    t2s.load_state_dict(saved["t2s"], assign=True)
    pipe.s2m.load_state_dict(saved["s2m"], assign=True)
    with torch.inference_mode(), amp():
        actual = t2s.get_submodule(first)(x)
    if not torch.equal(expected, actual):
        raise RuntimeError("calibrated checkpointの往復検査に失敗")
    print(f"calibration {condition}: checkpoint roundtrip OK", flush=True)
    return meta
