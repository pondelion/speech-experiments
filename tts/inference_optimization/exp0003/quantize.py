"""CPU動的INT8の変換範囲を明示する。"""
import warnings


def dynamic_int8(torch, model, exclude=()):
    names = {name for name, module in model.named_modules()
             if isinstance(module, torch.nn.Linear) and name not in exclude}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return torch.ao.quantization.quantize_dynamic(model, names, dtype=torch.qint8, inplace=False)


def weight_only(torch, model, bits, exclude=(), group_size=128, packing="tile_packed_to_4d"):
    from torchao.quantization import quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig
    config = Int8WeightOnlyConfig(set_inductor_config=False) if bits == 8 else Int4WeightOnlyConfig(group_size=group_size, set_inductor_config=False)
    if bits == 4:
        config.int4_packing_format = type(config.int4_packing_format)(packing)
    names, skipped = [], []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name in exclude or (bits == 4 and (module.in_features % group_size or module.out_features % 8)):
                skipped.append(name)
            else:
                names.append(name)
    quantize_(model, config, filter_fn=lambda module, name: name in names)
    checks = {}
    if bits == 4 and packing == "plain":
        from repack import repack_model
        checks = repack_model(torch, model, names)
    return {"quantized_layers": names, "excluded_layers": skipped,
            "bits": bits, "group_size": group_size if bits == 4 else None,
            "requested_packing": packing if bits == 4 else "Int8Tensor",
            "packing": "tile_packed_to_4d" if checks else (packing if bits == 4 else "Int8Tensor"),
            "repack_checks": checks}
