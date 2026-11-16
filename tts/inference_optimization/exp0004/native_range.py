"""PyTorch動的INT8のweightと入力範囲0..127をONNXで再現する。"""

import json
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.ao.quantization import default_dynamic_qconfig

ROOT = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    report = {}
    for graph in ["encoder", "blocks", "velocity"]:
        filename = f"{graph}_int8.onnx"
        model = onnx.load(str(ROOT / "runs" / "pt8_int8" / filename))
        original = onnx.load(
            str(ROOT / "runs" / "optimized_fp32" / f"{graph}_fp32.onnx")
        )
        weights = {
            x.name: numpy_helper.to_array(x)
            for x in original.graph.initializer
        }
        replacements = {}
        for node in original.graph.node:
            if node.op_type != "MatMul" or node.input[1] not in weights:
                continue
            name = node.input[1]
            value = weights[name]
            if value.ndim != 2:
                continue
            tensor = torch.from_numpy(value.T.copy())
            observer = default_dynamic_qconfig.weight()
            observer(tensor)
            scale, zero = observer.calculate_qparams()
            quant = torch.quantize_per_tensor(
                tensor, float(scale), int(zero), torch.qint8
            )
            replacements[name + "_quantized"] = quant.int_repr().numpy().T
            replacements[name + "_scale"] = np.array(float(scale), np.float32)
            replacements[name + "_zero_point"] = np.array(int(zero), np.int8)
        total = changed = scales = scale_changed = 0
        max_relative_scale = 0.0
        for index, value in enumerate(model.graph.initializer):
            if value.name in replacements:
                before = numpy_helper.to_array(value)
                after = replacements[value.name]
                if value.name.endswith("_quantized"):
                    total += before.size
                    changed += int(np.count_nonzero(before != after))
                elif value.name.endswith("_scale"):
                    scales += 1
                    before, after = before.reshape(-1), after.reshape(-1)
                    scale_changed += int(not np.array_equal(before, after))
                    relative = np.abs(before - after) / np.maximum(
                        np.abs(before), 1e-30
                    )
                    max_relative_scale = max(
                        max_relative_scale, float(relative.max())
                    )
                model.graph.initializer[index].CopyFrom(
                    numpy_helper.from_array(
                        replacements[value.name].copy(), value.name
                    )
                )
        nodes = []
        for node in model.graph.node:
            if node.op_type != "DynamicQuantizeLinear":
                nodes.append(node)
                continue
            prefix = node.name + "_native_"

            def const(name, value, dtype=np.float64):
                key = prefix + name
                model.graph.initializer.append(
                    numpy_helper.from_array(np.array(value, dtype=dtype), key)
                )
                return key

            def op(kind, inputs, label, **attributes):
                key = prefix + label
                nodes.append(
                    helper.make_node(
                        kind, inputs, [key], name=key, **attributes
                    )
                )
                return key

            zero, limit = const("zero", 0.0), const("limit", 127.0)
            zfloat = const("zfloat", 0.0, np.float32)
            low = op("ReduceMin", [node.input[0]], "min", keepdims=0)
            high = op("ReduceMax", [node.input[0]], "max", keepdims=0)
            low = op("Min", [low, zfloat], "min0")
            high = op("Max", [high, zfloat], "max0")
            ld = op("Cast", [low], "min_d", to=TensorProto.DOUBLE)
            hd = op("Cast", [high], "max_d", to=TensorProto.DOUBLE)
            span = op("Sub", [hd, ld], "span")
            scale = op("Div", [span, limit], "scale_raw")
            tiny = const("tiny", 1 / np.finfo(np.float32).max)
            exceptional = op("Less", [scale, tiny], "exceptional")
            scale = op(
                "Where",
                [exceptional, const("fallback", 0.1), scale],
                "scale_safe",
            )
            floor = const("floor", float(np.float32(6.1e-5)))
            adjusted = op("Max", [scale, floor], "scale_final")
            sf = op("Cast", [scale], "scale_f", to=TensorProto.FLOAT)
            af = op("Cast", [adjusted], "adjusted_f", to=TensorProto.FLOAT)
            amplifier = op("Div", [af, sf], "amplifier")
            low = op("Mul", [low, amplifier], "min_adjusted")
            high = op("Mul", [high, amplifier], "max_adjusted")
            low = op("Cast", [low], "lo_d", to=TensorProto.DOUBLE)
            high = op("Cast", [high], "hi_d", to=TensorProto.DOUBLE)
            low = op("Div", [low, adjusted], "lo_scaled")
            high = op("Div", [high, adjusted], "hi_scaled")
            from_low = op("Sub", [zero, low], "from_low")
            from_high = op("Sub", [limit, high], "from_high")
            abs_low = op("Abs", [low], "abs_low")
            abs_high = op("Abs", [high], "abs_high")
            err_low = op("Sub", [zero, abs_low], "err_low")
            err_high = op("Sub", [limit, abs_high], "err_high")
            choose = op("Less", [err_low, err_high], "choose")
            zp = op("Where", [choose, from_low, from_high], "zp")
            zp = op("Clip", [zp, zero, limit], "zp_clipped")
            zp = op("Round", [zp], "zp_rounded")
            nodes += [
                helper.make_node(
                    "Cast", [zp], [node.output[2]], to=TensorProto.UINT8
                ),
                helper.make_node("Identity", [af], [node.output[1]]),
                helper.make_node(
                    "QuantizeLinear",
                    [node.input[0], node.output[1], node.output[2]],
                    [node.output[0]],
                ),
            ]
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        onnx.checker.check_model(model)
        path = ROOT / "runs" / "native7_int8" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        report[graph] = {
            "weight_values": total,
            "changed_weight_values": changed,
            "changed_weight_pct": 100 * changed / total,
            "scale_tensors": scales,
            "changed_scale_tensors": scale_changed,
            "max_relative_scale_difference": max_relative_scale,
        }
        print("NATIVE_RANGE", graph, len(nodes), flush=True)
    (ROOT / "results" / "quantization_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
