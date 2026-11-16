"""教師の読み込み、層選択によるstudent作成、重みの保存。"""

import copy
import io

import torch

from .paths import CHECKPOINTS, benchmark_module

FROZEN = ("encoder", "enc_emb", "duration", "align_proj")


def load_teacher(device):
    bench = benchmark_module()
    _, _, _, load_t2s, _, _ = bench.import_models()
    model, vocab, method = load_t2s(CHECKPOINTS["t2s"], device)
    if method != "hybrid" or model.duration_model != "nar":
        raise ValueError("Unexpected teacher method or duration predictor")
    model.eval().requires_grad_(False)
    checkpoint = torch.load(
        CHECKPOINTS["t2s"], map_location="cpu", weights_only=False
    )
    return model, vocab, checkpoint


def make_student(teacher, layers):
    """教師の先頭から末尾まで均等に層を選び、他の形状は維持する。"""
    count = len(teacher.blocks)
    if not 1 <= layers < count:
        raise ValueError("Student layers must be positive and below teacher")
    indices = (
        [count // 2]
        if layers == 1
        else [round(i * (count - 1) / (layers - 1)) for i in range(layers)]
    )
    student = copy.deepcopy(teacher)
    student.blocks = torch.nn.ModuleList([student.blocks[i] for i in indices])
    student.requires_grad_(True)
    for name in FROZEN:
        getattr(student, name).requires_grad_(False)
    return student, indices


def training_mode(student):
    """固定したencoder等のdropoutも固定する。"""
    student.train()
    for name in FROZEN:
        getattr(student, name).eval()


def parameter_inventory(model):
    groups = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".", 1)[0]
        groups[prefix] = groups.get(prefix, 0) + parameter.numel()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "by_module": groups,
        "state_dict_bytes": buffer.tell(),
    }


def save_student(path, student, source, indices, metadata):
    checkpoint = {
        "model": {
            k: v.detach().cpu() for k, v in student.state_dict().items()
        },
        "config": {**source["config"], "layers": len(indices)},
        "method": source["method"],
        "vocab": source["vocab"],
        "n_codes": source["n_codes"],
        "compression": {"teacher_layers": indices, **metadata},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def reload_student(path, device):
    bench = benchmark_module()
    _, _, _, load_t2s, _, _ = bench.import_models()
    return load_t2s(path, device)[0]
