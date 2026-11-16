"""exp0006の実行窓口。計算処理はcommon配下へ分離する。"""

import argparse
import fcntl

from .common.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_a = commands.add_parser("stage-a", help="基準・学習経路の検証")
    stage_a.add_argument("--device", choices=["cpu", "cuda"], required=True)
    stage_a.add_argument("--repeats", type=int, default=10)
    train = commands.add_parser("train", help="層数縮小と蒸留の固定予算比較")
    train.add_argument("--device", default="cuda", choices=["cuda"])
    train.add_argument("--layers", type=int, required=True, choices=[6, 4])
    train.add_argument("--mode", choices=["plain", "kd"], required=True)
    train.add_argument("--steps", type=int, default=1000)
    train.add_argument("--max-tokens", type=int, default=4096)
    train.add_argument("--eval-every", type=int, default=250)
    train.add_argument("--lr", type=float, default=3e-5)
    train.add_argument("--seed", type=int, default=0)
    evaluate = commands.add_parser("evaluate", help="速度・CER・代表MP3")
    evaluate.add_argument("--device", choices=["cpu", "cuda"], required=True)
    evaluate.add_argument("--layers", type=int, choices=[6, 4], required=True)
    evaluate.add_argument("--mode", choices=["plain", "kd"], required=True)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--repeats", type=int, default=10)
    commands.add_parser("summarize", help="実測結果をREADMEに反映")
    args = parser.parse_args()
    if args.command in ("stage-a", "evaluate") and args.repeats < 1:
        parser.error("repeats must be positive")
    lock = ROOT.parent / "exp0004" / "runs" / "locks" / "phase.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if args.command == "stage-a":
            from .common.audit import run_stage_a

            run_stage_a(args)
        elif args.command == "train":
            if min(args.steps, args.max_tokens, args.eval_every) < 1:
                parser.error("Training counts must be positive")
            from .common.training import train_student

            train_student(args)
        elif args.command == "evaluate":
            from .common.evaluation import evaluate_condition

            evaluate_condition(args)
        elif args.command == "summarize":
            from .common.reporting import summarize

            summarize()


if __name__ == "__main__":
    main()
