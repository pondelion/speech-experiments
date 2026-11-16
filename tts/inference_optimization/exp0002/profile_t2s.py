"""ハイブリッドT2Sの区間を計測し、計測前後の生成ID一致を確認する。"""

import argparse
import ast
import fcntl
import hashlib
import inspect
import json
import statistics
import sys
import textwrap
import time
import types
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Tracker:
    def __init__(self, torch, device):
        self.torch = torch
        self.cuda = device == "cuda"
        self.events = []
        self.stack = []
        self.pool = []
        if self.cuda:
            for _ in range(4096):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                end.record()
                self.pool.append((begin, end))
            torch.cuda.synchronize()

    @contextmanager
    def section(self, name):
        start = time.perf_counter()
        record = {"name": name, "children": []}
        if self.stack:
            self.stack[-1]["children"].append(record)
        self.events.append(record)
        self.stack.append(record)
        if self.cuda:
            record["begin"], record["end"] = self.pool[len(self.events) - 1]
            record["begin"].record()
        try:
            yield
        finally:
            if self.cuda:
                record["end"].record()
            record["cpu_ms"] = 1000 * (time.perf_counter() - start)
            self.stack.pop()

    def summarize(self):
        if self.cuda:
            self.torch.cuda.synchronize()
        for event in self.events:
            event["stream_ms"] = (
                event["begin"].elapsed_time(event["end"])
                if self.cuda
                else event["cpu_ms"]
            )
        result = defaultdict(lambda: defaultdict(float))
        for event in self.events:
            target = result[event["name"]]
            target["calls"] += 1
            for axis in ["cpu", "stream"]:
                value = event[f"{axis}_ms"]
                target[f"{axis}_inclusive_ms"] += value
                children = sum(c[f"{axis}_ms"] for c in event["children"])
                target[f"{axis}_self_ms"] += max(0, value - children)
        return dict(result)


def scoped(statement, label):
    return ast.With(
        items=[
            ast.withitem(
                ast.Call(
                    func=ast.Name("_t2s_section", ast.Load()),
                    args=[ast.Constant(label)],
                    keywords=[],
                )
            )
        ],
        body=statement if isinstance(statement, list) else [statement],
    )


def instrument_loop(statements):
    result = []
    labels = {
        "t": "loop_input",
        "xt": "loop_input",
        "std": "cfg_mix",
        "seq": "token_update",
        "n_next": "remask_schedule",
        "prob": "confidence_softmax",
        "unsure": "confidence_score",
        "rank": "remask_rank",
        "masked": "mask_update",
    }
    for statement in statements:
        label = None
        if isinstance(statement, ast.Assign):
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                label = labels.get(target.id)
                if target.id == "h" and not (
                    isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Attribute)
                    and statement.value.func.attr == "hidden"
                ):
                    label = "cfg_mix"
            elif isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name):
                    if target.value.id == "logits":
                        label = "topk_filter"
        elif isinstance(statement, ast.If):
            test = ast.unparse(statement.test)
            if "final_argmax" in test:
                label = "sampling"
            elif "n_next" in test:
                label = "stop_check"
            else:
                statement.body = instrument_loop(statement.body)
                statement.orelse = instrument_loop(statement.orelse)
        result.append(scoped(statement, label) if label else statement)
    return result


def measured_function(function, tracker, hidden=False):
    original = inspect.unwrap(function)
    if inspect.ismethod(original):
        original = original.__func__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    definition = tree.body[0]
    definition.decorator_list = []
    if hidden:
        body = []
        index = 0
        labels = {
            "drops": ("hidden_text_setup", 2),
            "seqs": ("sequence_assembly", 2),
            "pad": ("padding_mask", 1),
            "t_emb": ("time_condition", 1),
            "audio": ("output_assembly", 3),
        }
        while index < len(definition.body):
            statement = definition.body[index]
            if isinstance(statement, ast.Assign):
                target = statement.targets[0]
                name = target.id if isinstance(target, ast.Name) else ""
                if name in labels:
                    label, count = labels[name]
                    body.append(
                        scoped(definition.body[index : index + count], label)
                    )
                    index += count
                    continue
            if isinstance(statement, ast.If):
                statement = scoped(statement, "audio_condition_add")
            body.append(statement)
            index += 1
        definition.body = body
    else:
        for statement in definition.body:
            if isinstance(statement, ast.For):
                statement.body = instrument_loop(statement.body)
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace["_t2s_section"] = tracker.section
    exec(compile(tree, "instrumented_masked_t2s", "exec"), namespace)
    return namespace[definition.name]


@contextmanager
def instrument(torch, model, tracker):
    restores = []

    def wrap(obj, attribute, label):
        had_attribute = attribute in obj.__dict__
        previous = obj.__dict__.get(attribute)
        original = getattr(obj, attribute)

        def call(*args, **kwargs):
            name = label
            if label == "encoder_total":
                parent_names = [e["name"] for e in tracker.stack]
                name = (
                    "duration_encoder"
                    if "duration_total" in parent_names
                    else "generate_encoder"
                )
            with tracker.section(name):
                return original(*args, **kwargs)

        setattr(obj, attribute, call)
        restores.append((obj, attribute, had_attribute, previous))

    masked = next(c for c in type(model).__mro__ if c.__name__ == "MaskedT2S")
    original_masked = masked.generate
    measured_masked = measured_function(original_masked, tracker)
    previous_hidden = model.__dict__.get("hidden")
    had_hidden = "hidden" in model.__dict__
    compiled_hidden = measured_function(model.hidden, tracker, hidden=True)
    restores.append((model, "hidden", had_hidden, previous_hidden))
    model.hidden = types.MethodType(compiled_hidden, model)

    def generate(instance, *args, **kwargs):
        with tracker.section("masked_generate"):
            return measured_masked(instance, *args, **kwargs)

    class Blocks(torch.nn.ModuleList):
        def __iter__(self):
            with tracker.section("transformer_blocks"):
                yield from super().__iter__()

    original_blocks = model.blocks
    model.blocks = Blocks(list(original_blocks))
    masked.generate = generate
    for attribute, label in [
        ("predict_durations", "duration_total"),
        ("generate", "hybrid_generate"),
        ("encode", "encoder_total"),
        ("align", "alignment"),
        ("hidden", "hidden_total"),
        ("embed_codes", "audio_embedding"),
    ]:
        wrap(model, attribute, label)
    for attribute, label in [
        ("encoder", "encoder_nn"),
        ("duration", "duration_head"),
        ("enc_emb", "encoder_lookup"),
        ("text_emb", "text_lookup"),
        ("time_mlp", "time_embedding_nn"),
        ("norm", "hidden_norm"),
        ("head", "vocab_projection"),
    ]:
        wrap(getattr(model, attribute), "forward", label)
    try:
        yield
    finally:
        for obj, attribute, had_attribute, previous in reversed(restores):
            if had_attribute:
                setattr(obj, attribute, previous)
            else:
                delattr(obj, attribute)
        model.blocks = original_blocks
        masked.generate = original_masked


def main():
    sys.path.insert(0, str(ROOT.parent / "exp0001"))
    sys.path.insert(0, str(ROOT.parent / "exp0004"))
    from benchmark import build_models, environment, load_inputs, one_run
    from runtime import prepare

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--case", default="all")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = ROOT / "results" / "t2s_profile"
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{args.device}_{args.config}_{args.case}.json"
    if path.exists():
        print("SKIP", path.name)
        return
    locks = ROOT.parent / "exp0004" / "runs" / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = (locks / "phase.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch, pipe, model, vocab, seed = build_models(args.device)
    metadata, model = prepare(torch, pipe, model, args.device, args.config)
    source = inspect.getsource(
        inspect.unwrap(
            next(
                c for c in type(model).__mro__ if c.__name__ == "MaskedT2S"
            ).generate
        )
    )
    result = {
        "environment": environment(torch, args),
        "runtime": metadata,
        "model": "hubert_ja_k256/tts",
        "method": "hybrid",
        "duration_model": model.duration_model,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "repeats": args.repeats,
        "measurement_version": 2,
        "cases": {},
        "gpu_clock": "CUDA stream elapsed, including host dispatch gaps",
        "gpu_events_preallocated": args.device == "cuda",
    }
    for item in load_inputs(args.case):
        text = item["text"]
        one_run(
            torch, pipe, model, vocab, seed, text, args.device, 0, "float32"
        )
        baseline, hashes = [], []
        for index in range(args.repeats):
            times, shape = one_run(
                torch,
                pipe,
                model,
                vocab,
                seed,
                text,
                args.device,
                index + 1,
                "float32",
            )
            baseline.append(times)
            hashes.append(shape["codes_sha256"])
        profiles = []
        paired_reference_s = []
        full_pipeline_hash_matches = []
        for index in range(args.repeats):
            tracker = Tracker(torch, args.device)
            seed(index + 1)
            ids = [torch.tensor(vocab.encode(text))]
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                reference = model.generate(
                    ids, durations=model.predict_durations(ids)
                )
            if args.device == "cuda":
                torch.cuda.synchronize()
            paired_reference_s.append(time.perf_counter() - start)
            reference_hash = hashlib.sha256(
                reference[0].numpy().tobytes()
            ).hexdigest()
            full_pipeline_hash_matches.append(reference_hash == hashes[index])
            seed(index + 1)
            ids = [torch.tensor(vocab.encode(text))]
            if args.device == "cuda":
                torch.cuda.synchronize()
            with torch.inference_mode(), instrument(torch, model, tracker):
                with tracker.section("t2s_total"):
                    durations = model.predict_durations(ids)
                    codes = model.generate(ids, durations=durations)
            digest = hashlib.sha256(codes[0].numpy().tobytes()).hexdigest()
            if digest != reference_hash:
                raise RuntimeError(
                    f"Instrumentation changed IDs: {item['id']} "
                    f"seed={index + 1}"
                )
            profiles.append(tracker.summarize())
        names = profiles[0].keys()
        aggregate = {
            name: {
                key: statistics.median(p[name][key] for p in profiles)
                for key in profiles[0][name]
            }
            for name in names
        }
        result["cases"][item["id"]] = {
            "baseline_runs": baseline,
            "profiles": profiles,
            "median_profile": aggregate,
            "generated_ids_identical": True,
            "paired_reference_s": paired_reference_s,
            "full_pipeline_hash_matches": full_pipeline_hash_matches,
            "baseline_median_s": {
                key: statistics.median(r[key] for r in baseline)
                for key in baseline[0]
            },
        }
        path.with_suffix(".partial.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(item["id"], aggregate["t2s_total"], flush=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.with_suffix(".partial.json").unlink(missing_ok=True)
    print("DONE", path.name, flush=True)


if __name__ == "__main__":
    main()
