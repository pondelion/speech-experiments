"""蒸留有無で共有する学習stepと保存・再ロードの動作確認。"""

import json
import time

import torch

from .data import select_batch
from .distillation import loss_for
from .models import (
    FROZEN,
    make_student,
    reload_student,
    save_student,
    training_mode,
)
from .paths import RUNS, relative


def optimization_step(student, teacher, data, batch, optimizer, kd_weight):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=student.device.type == "cuda"
    ):
        loss, parts = loss_for(student, teacher, data, batch, kd_weight)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad],
        1.0,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(norm), **parts}


def smoke_training(teacher, source, data, device):
    """本学習の代わりではなく、実データで学習経路を検証する。"""
    results = {}
    batch = select_batch(data["train"])
    for mode, weight in (("plain", 0.0), ("kd", 1.0)):
        torch.manual_seed(0)
        student, indices = make_student(teacher, 6)
        training_mode(student)
        before = {n: p.detach().clone() for n, p in student.named_parameters()}
        optimizer = torch.optim.AdamW(
            [p for p in student.parameters() if p.requires_grad], lr=1e-5
        )
        torch.manual_seed(42)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        history = [
            optimization_step(
                student, teacher, data["train"], batch, optimizer, weight
            )
            for _ in range(2)
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        frozen_equal = all(
            torch.equal(before[n], p)
            for n, p in student.named_parameters()
            if n.split(".")[0] in FROZEN
        )
        changed = any(
            not torch.equal(before[n], p)
            for n, p in student.named_parameters()
            if n.startswith("blocks.")
        )
        assert frozen_equal and changed
        student.eval()
        path = RUNS / "smoke" / device / mode / "t2s.pt"
        save_student(
            path,
            student,
            source,
            indices,
            {"smoke_only": True, "steps": 2, "kd_weight": weight},
        )
        restored = reload_student(path, device)
        state_equal = all(
            torch.equal(v, restored.state_dict()[k])
            for k, v in student.state_dict().items()
        )
        text = [data["val"]["texts"][0]]
        with torch.inference_mode():
            torch.manual_seed(123)
            generated = student.generate(text)
            torch.manual_seed(123)
            reloaded = restored.generate(text)
        ids_equal = all(torch.equal(a, b) for a, b in zip(generated, reloaded))
        assert state_equal and ids_equal
        results[mode] = {
            "steps": 2,
            "batch_size": len(batch),
            "selected_teacher_layers": indices,
            "kd_weight": weight,
            "history": history,
            "training_s": elapsed,
            "frozen_parameters_unchanged": frozen_equal,
            "generation_blocks_updated": changed,
            "reload_state_identical": state_equal,
            "reload_generated_ids_identical": ids_equal,
            "checkpoint": relative(path),
            "peak_cuda_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if device == "cuda"
                else None
            ),
        }
        print(f"SMOKE {device}/{mode} passed", flush=True)
        del student, restored, optimizer, before
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


@torch.no_grad()
def validation_ce(student, teacher, data, max_tokens):
    """蒸留有無で共通の正解トークンCEをcheckpoint選択に使う。"""
    from .data import batches

    devices = (
        [student.device.index or 0] if student.device.type == "cuda" else []
    )
    student.eval()
    total, count = 0.0, 0
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(123)
        for batch in batches(data, max_tokens, shuffle=False):
            _, parts = loss_for(student, teacher, data, batch)
            total += parts["ce"] * parts["masked_positions"]
            count += parts["masked_positions"]
    training_mode(student)
    return total / count


def train_student(args):
    """固定予算で通常学習と蒸留を比較し、共通検証CE最小を保存。"""
    import hashlib
    import math

    from .data import batches, load_data
    from .models import load_teacher, parameter_inventory
    from .paths import CHECKPOINTS, RESULTS, digest, write_json

    if args.device != "cuda":
        raise ValueError("Full training currently uses CUDA")
    torch.set_num_threads(8)
    condition = f"t2s_l{args.layers}_{args.mode}_s{args.seed}"
    directory = RUNS / condition
    target = RESULTS / "training" / f"{condition}.json"
    if target.exists():
        raise FileExistsError(f"Completed training exists: {condition}")
    teacher, vocab, source = load_teacher(args.device)
    data, audit = load_data(vocab)
    torch.manual_seed(args.seed)
    student, indices = make_student(teacher, args.layers)
    initial_hash = hashlib.sha256()
    for name, parameter in student.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(parameter.cpu().numpy().tobytes())
    metadata = {
        "condition": condition,
        "layers": args.layers,
        "mode": args.mode,
        "seed": args.seed,
        "steps": args.steps,
        "max_tokens": args.max_tokens,
        "lr": args.lr,
        "eval_every": args.eval_every,
        "kd_weight": 1.0 if args.mode == "kd" else 0.0,
        "temperature": 1.0,
        "initial_state_sha256": initial_hash.hexdigest(),
        "teacher_sha256": digest(CHECKPOINTS["t2s"]),
        "data_identity": audit["identity"],
        "data_ids_sha256": {
            group: audit["groups"][group]["ids_sha256"]
            for group in ("train", "val")
        },
        "selected_teacher_layers": indices,
        "frozen_modules": list(FROZEN),
        "inventory": parameter_inventory(student),
        "checkpoint_selection": "minimum validation CE over masked tokens",
        "training_precision": "bfloat16 autocast; FP32 master weights",
    }
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "config.json", metadata)
    training_mode(student)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    best = validation_ce(student, teacher, data["val"], args.max_tokens)
    history = [{"step": 0, "validation_ce": best}]
    save_student(directory / "t2s.pt", student, source, indices, metadata)
    torch.manual_seed(args.seed + 100)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    step, epoch, examples, best_step = 0, 0, 0, 0
    log_path = directory / "log.jsonl"
    while step < args.steps:
        for batch in batches(data["train"], args.max_tokens, epoch):
            step += 1
            warmup = min(50, max(1, args.steps // 10))
            factor = (
                min(step / warmup, 1.0)
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * max(0, step - warmup)
                        / max(1, args.steps - warmup)
                    )
                )
            )
            optimizer.param_groups[0]["lr"] = args.lr * factor
            parts = optimization_step(
                student,
                teacher,
                data["train"],
                batch,
                optimizer,
                metadata["kd_weight"],
            )
            examples += len(batch)
            row = {"step": step, "batch_size": len(batch), **parts}
            with log_path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                ce = validation_ce(
                    student, teacher, data["val"], args.max_tokens
                )
                if ce < best:
                    best, best_step = ce, step
                    save_student(
                        directory / "t2s.pt",
                        student,
                        source,
                        indices,
                        {**metadata, "best_step": best_step},
                    )
                entry = {"step": step, "validation_ce": ce, **parts}
                history.append(entry)
                write_json(directory / "progress.json", history)
                print(
                    f"TRAIN {condition} {step}/{args.steps} "
                    f"val_ce={ce:.4f} best={best:.4f}",
                    flush=True,
                )
            if step >= args.steps:
                break
        epoch += 1
    torch.cuda.synchronize()
    metadata.update(
        {
            "complete": True,
            "training_s": time.perf_counter() - start,
            "processed_examples": examples,
            "best_step": best_step,
            "best_validation_ce": best,
            "history": history,
            "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated()
            / 2**20,
            "checkpoint": relative(directory / "t2s.pt"),
            "checkpoint_sha256": digest(directory / "t2s.pt"),
        }
    )
    write_json(target, metadata)
    print(f"TRAIN COMPLETE {condition}", flush=True)
