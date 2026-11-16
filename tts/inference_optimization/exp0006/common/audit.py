"""段階A: 重み・データ分割・学習経路・速度の実行検査。"""

import json
import platform
from importlib.metadata import version

import torch

from .data import load_data
from .evaluation import measure, reference_quality
from .models import load_teacher, make_student, parameter_inventory
from .paths import CHECKPOINTS, RESULTS, digest, relative, write_json
from .training import smoke_training


def run_stage_a(args):
    output = RESULTS / "stage_a" / f"{args.device}.json"
    if output.exists():
        raise FileExistsError(f"Completed result exists: {relative(output)}")
    torch.set_num_threads(8)
    teacher, vocab, source = load_teacher(args.device)
    torch.manual_seed(0)
    data, audit = load_data(vocab)
    write_json(RESULTS / "data_audit.json", audit)
    inventories = {"teacher": parameter_inventory(teacher)}
    for layers in (6, 4):
        student, indices = make_student(teacher, layers)
        inventories[f"layers{layers}"] = {
            **parameter_inventory(student),
            "selected_teacher_layers": indices,
        }
        del student
    result = {
        "stage": "A",
        "complete": False,
        "device": args.device,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "onnxruntime-gpu")
            },
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name()
            if args.device == "cuda"
            else None,
            "threads": torch.get_num_threads(),
        },
        "checkpoints": {
            name: {
                "path": relative(path),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for name, path in CHECKPOINTS.items()
        },
        "teacher_config": source["config"],
        "method": source["method"],
        "inventory": inventories,
    }
    result["smoke"] = smoke_training(teacher, source, data, args.device)
    result["benchmark"] = measure(teacher, vocab, args.device, args.repeats)
    result["quality_reference"] = reference_quality(args.device)
    result["complete"] = True
    write_json(output, result)
    print(
        json.dumps({"saved": relative(output), "complete": True}), flush=True
    )
