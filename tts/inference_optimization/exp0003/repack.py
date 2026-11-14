"""MSLK signed INT4を、再量子化せずPyTorch tinygemm配置へ変換する。"""
def repack_plain_int4(torch, weight):
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
        Int4TilePackedTo4dTensor, pack_tinygemm_scales_and_zeros,
    )
    if weight.scale.dtype != torch.bfloat16:
        raise ValueError("scaleを丸めず変換するためBF16 scaleが必要")
    n, k = weight.shape
    group = weight.block_size[-1]
    raw = weight.qdata.to(torch.uint8)
    # MSLK: 偶数列がlow nibble、signed [-8,7]。tinygemm: 偶数列がhigh、unsigned [0,15]。
    signed_bits = torch.stack([raw & 15, raw >> 4], dim=-1).reshape(n, k)
    unsigned = signed_bits ^ 8
    padded_k = ((k + 1023) // 1024) * 1024
    q = torch.full((n, padded_k), 8, dtype=torch.uint8, device=raw.device)
    q[:, :k] = unsigned
    packed_bytes = (q[:, ::2] << 4) | q[:, 1::2]
    packed = torch.ops.aten._convert_weight_to_int4pack(packed_bytes.contiguous(), 8)
    scale = torch.ones(n, padded_k // group, dtype=torch.bfloat16, device=raw.device)
    zero = torch.zeros_like(scale)
    scale[:, :k//group] = weight.scale.t()
    zero[:, :k//group] = weight.zero_point.t()
    assert torch.equal(scale[:, :k//group], weight.scale.t())
    assert torch.equal(zero[:, :k//group], weight.zero_point.t())
    result = Int4TilePackedTo4dTensor(qdata=packed,
        scale_and_zero=pack_tinygemm_scales_and_zeros(scale, zero, scale.dtype),
        block_size=list(weight.block_size), shape=weight.shape, act_pre_scale=weight.act_pre_scale)
    # GPU kernelが表すdense weightをidentity入力で検証する。再量子化・scale丸めは行っていない。
    signed = signed_bits.float() - 16 * (signed_bits >= 8)
    expected = signed * weight.scale.t().float().repeat_interleave(group, 1)
    expected += weight.zero_point.t().float().repeat_interleave(group, 1)
    eye = torch.eye(k, device=raw.device, dtype=torch.bfloat16)
    actual = torch.nn.functional.linear(eye, result).t().float()
    relative = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    if relative > 0.01:
        raise RuntimeError(f"INT4 packing変換誤差が大きい: {relative}")
    return result, {"qparams_preserved": True, "no_requantization": True,
                    "dense_weight_relative_l2": relative,
                    "dense_weight_max_abs": (actual-expected).abs().max().item()}


def repack_model(torch, model, names):
    checks = {}
    with torch.no_grad():
        for name in names:
            layer = model.get_submodule(name)
            new_weight, checks[name] = repack_plain_int4(torch, layer.weight)
            layer.weight = torch.nn.Parameter(new_weight, requires_grad=False)
    return checks
