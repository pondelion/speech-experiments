"""評価用の音声を合成する(指標は metrics.py)。1 つの条件を 1 つのプロセスで(続けてモデルを替えると GPU のエラーで落ちることがあるため)。

条件(--cond):
    spk     この exp の T2S・S2M(--t2s・--s2m)+ 多話者で追加学習した Vocos(--vocoder)。参照音声の声で読む
    phase_a 比べる相手: フェーズ A の一番良い構成(exp0005: t2s_f0aux_w2 + s2m_f0cond + JSUT の Vocos)。参照音声は使わない(JSUT の声)
評価する話者: 学習に使わない 10 人(unseen)と、学習に使った話者から選んだ 10 人(seen)(results/split.json)。
参照音声: 話者ごとに 1 つ。unseen はその人の nonpara30(prompt)、seen はその人の学習データから、ID の順で最初の 3〜8 秒の発話
文: テスト文 25 文(その話者の本物の録音がある)と実用文 55 文。乱数の種 0・1・2
生成(spk): フェーズ A の一番良い構成と同じ設定(T2S: 50 回、温度 0.9、上位 2%、CFG 0。S2M: Euler 25 step、CFG 0.7)。
    T2S には参照音声の話者ベクトル(CAM++)を与える。S2M は参照音声のトークン・F0 を前につなげ、参照の Mel をプロンプトにして作り、
    参照の区間を取り除く(公式の推論と同じ)

使い方:
    python synth.py --cond spk --t2s t2s_finetune --s2m s2m_finetune --vocoder vocos_multi --name finetune
    python synth.py --cond phase_a --name phase_a
    python synth.py --cond spk --t2s t2s_finetune --s2m s2m_finetune --vocoder vocos_multi --name smoke --limit 2 --seeds 1
出力: runs/eval/{name}/seed{s}/{test,practical}/{話者}/{文の ID}.wav と runs/eval/{name}/refs.json(使った参照音声)
"""
import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

import common
from common.data import list_all, load_split
from common.f0norm import mean_lf0, normalize, raw_targets
from common.s2m_spk import SpeakerS2M
from common.t2s_spk import SpeakerHybridT2S
from exp0001.common.utils import load_codec
from exp0005.common.s2m_f0 import FPT, F0Condition
from exp0005.train_t2s import f0_targets
from lib_common.audio import SR
from lib_common.eval_sets import load_eval_texts
from lib_common.neural_vocoder import load_vocoder
from lib_common.text import TextVocab
from lib_common.utils import Logger, seed_everything

E3, E4, E5 = (common.EXP_DIR.parent / e / "runs" for e in ("exp0003", "exp0004", "exp0005"))
T2S_KW = {"n_steps": 50, "temperature": 0.9, "filter_thres": 0.98, "cfg": 0.0}
S2M_KW = {"n_steps": 25, "cfg": 0.7}
BATCH = 16


def pick_refs(utts: list, mels: dict) -> dict:
    """評価する話者ごとの参照音声(ID の順で最初の 3〜8 秒の発話)。"""
    sp = load_split()
    refs = {}
    for group, split_name in [("unseen", "prompt"), ("seen", "train")]:
        for s in sp[f"{group}_speakers" if group == "unseen" else "seen_eval_speakers"]:
            cands = sorted((u for u in utts if u.speaker == s and u.split == split_name), key=lambda u: u.id)
            refs[s] = next(u for u in cands if 3.0 <= len(mels[u.id]) * 256 / SR <= 8.0)
    return refs


def external_ref(path: str, name: str, mels: dict, tokens: dict, f0: dict, spk: dict):
    """データセットに無い参照音声の Mel・トークン・F0・話者ベクトルを作り、各 dict に ID name で入れる(prepare_data.py と同じ手順)。"""
    from types import SimpleNamespace

    from common.speaker import CAMPlus
    from exp0001.common.utils import encode_all
    from exp0003.common.ssl import SSLFeatures
    from exp0003.common.utils import codec_inputs, load_ssl_codec
    from lib_common.audio import load_wave, wave_to_logmel
    from lib_common.prosody import f0 as pyin_f0

    import numpy as np
    u = SimpleNamespace(id=name, path=path, speaker=name)
    u.mel = wave_to_logmel(load_wave(path))
    mels[name] = u.mel.half()
    model, _, feat_norm, info = load_ssl_codec(E3 / "codec_hubert_ja_k256" / "codec.pt", "cuda")
    enc = SSLFeatures(info["ssl"], "cuda", info["layer"])
    tokens[name] = encode_all(model, codec_inputs([enc(path)], [u], feat_norm), "cuda")[0].short()
    del enc, model
    torch.cuda.empty_cache()
    fv, vv = pyin_f0(load_wave(path))
    f0[name] = (torch.from_numpy(np.nan_to_num(fv).astype(np.float16)), torch.from_numpy(vv))
    spk[name] = CAMPlus("cpu")(path)
    return u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=["spk", "phase_a"], required=True)
    parser.add_argument("--t2s", default="t2s_finetune")
    parser.add_argument("--s2m", default="s2m_finetune")
    parser.add_argument("--vocoder", default="vocos_multi")
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="動作確認用に、文の数を絞る")
    parser.add_argument("--ref_wav", default=None,
                        help="データセットに無い参照音声(例: VOICEVOX のずんだもん)。与えると、評価する話者の代わりにこの声 1 つで合成する")
    parser.add_argument("--ref_name", default="external", help="--ref_wav の話者の名前(出力のフォルダ名)")
    parser.add_argument("--out_root", default=None, help="評価の音声の保存先の runs/(既定は exp0101/runs。exp0102 は exp0102/runs)")
    parser.add_argument("--speakers", nargs="*", default=None,
                        help="作る話者を絞る(試聴用のサンプルを作り直すとき。話者ごと・文のまとまりごとに乱数の種を決め直すので、同じ種なら評価と同じ乱数になる)")
    args = parser.parse_args()
    torch.set_num_threads(4)
    R = common.RUNS_DIR
    out_dir = (Path(args.out_root) if args.out_root else R) / "eval" / args.name
    log = Logger(out_dir / "log.txt")
    log(f"args: {vars(args)}")
    utts = list_all()
    by_id = {u.id: u for u in utts}
    mels, tokens, f0, spk = (torch.load(R / f) for f in ("mels.pt", "tokens.pt", "f0.pt", "spk.pt"))
    sp = load_split()
    test_texts = sorted(sp["test_texts"].items())[: args.limit]
    practical = [(t.id, t.text) for t in load_eval_texts(["practical"])][: args.limit]
    codec, mel_norm = load_codec(E3 / "codec_hubert_ja_k256" / "codec.pt", "cuda")
    if args.ref_wav:  # データセットに無い参照音声: Mel・トークン・F0・話者ベクトルを prepare_data.py と同じ手順でここで作る
        refs = {args.ref_name: external_ref(args.ref_wav, args.ref_name, mels, tokens, f0, spk)}
    else:
        refs = pick_refs(utts, mels)
    (out_dir / "refs.json").write_text(json.dumps({s: u.id for s, u in refs.items()}, indent=1))

    t_opt, s_opt = {}, {}
    if args.cond == "spk":
        ck = torch.load(R / args.t2s / "t2s.pt", map_location="cpu")
        vocab = TextVocab.from_state_dict(ck["vocab"])
        sck = torch.load(R / args.s2m / "s2m.pt", map_location="cpu")
        t_opt, s_opt = ck.get("options", {}), sck.get("options", {})  # exp0102 の条件(無ければ exp0101 の基礎の構成)
        if t_opt.get("ref_encoder"):  # 話者の条件を参照音声の Mel から作る T2S(common/t2s_ref.py)
            from common.t2s_ref import RefEncoderT2S
            t2s = RefEncoderT2S(len(vocab), ck["n_codes"], **ck["config"])
        else:
            t2s = SpeakerHybridT2S(len(vocab), ck["n_codes"], **ck["config"])
        t2s.load_state_dict(ck["model"])
        assert t_opt.get("f0_norm", "jsut") == s_opt.get("f0_norm", "jsut"), "T2S と S2M の F0 の正規化が違う"
        s2m = SpeakerS2M(sck["cond_in_dim"], use_prompt=not s_opt.get("no_prompt", False), **sck["config"])
        s2m.load_state_dict(sck["s2m"])
        voc = load_vocoder(R / args.vocoder / "vocoder.pt", "cuda")
        log(f"options: T2S {t_opt}, S2M {s_opt}")
    else:
        from exp0002.common.utils import load_t2s
        from exp0001.common.pipeline import load_s2m
        t2s, vocab, _ = load_t2s(E5 / "t2s_f0aux_w2" / "t2s.pt", "cuda")
        sck = torch.load(E5 / "s2m_f0cond" / "s2m.pt", map_location="cpu")
        s2m = load_s2m(E3 / "s2m_hubert_ja_k256" / "s2m.pt", "cuda")
        s2m.load_state_dict(sck["s2m"])
        voc = load_vocoder(E4 / "vocos" / "vocoder.pt", "cuda")
    t2s, s2m = t2s.cuda().eval(), s2m.cuda().eval()
    f0cond = F0Condition(codec.hidden_dim).cuda().eval()
    f0cond.load_state_dict(sck["f0cond"])
    stats = sck["f0_stats"]

    @torch.no_grad()
    def synth(texts: list[str], ref) -> list:
        """テキストのリストを、参照音声 ref の声で合成した波形のリスト(phase_a は ref を使わない)。"""
        waves = []
        for s in range(0, len(texts), BATCH):
            ids = [torch.tensor(vocab.encode(t)) for t in texts[s: s + BATCH]]
            if args.cond == "phase_a":
                d = t2s.predict_durations(ids)
                codes = t2s.generate(ids, durations=d, **T2S_KW)
                fs = t2s.predict_f0(ids, [c.cuda() for c in codes], d)
                hidden = [codec.decode_hidden(c.cuda()[None])[0] + f0cond(f.cuda().float()) for c, f in zip(codes, fs)]
                for h, c in zip(hidden, codes):
                    mask = torch.ones(1, len(c) * FPT, dtype=torch.bool, device="cuda")
                    m = s2m.generate(h[None], mask, **S2M_KW)[0]
                    waves.append(voc.generate(mel_norm.denormalize(m)[None])[0].cpu().numpy())
                continue
            if t_opt.get("ref_encoder"):  # 参照音声の Mel(S2M と同じ正規化)を発話の数だけ並べる
                sv = [mel_norm.normalize(mels[ref.id].float())] * len(ids)
            else:
                sv = spk[ref.id][None].expand(len(ids), -1)
                sv = torch.zeros_like(sv) if t_opt.get("no_spk") else sv
            with t2s.speaker(sv):
                d = t2s.predict_durations(ids)
                codes = t2s.generate(ids, durations=d, **T2S_KW)
                fs = t2s.predict_f0(ids, [c.cuda() for c in codes], d)
            rc = tokens[ref.id].long()
            if s_opt.get("f0_norm") == "ref_mean":  # 参照音声の log F0 の平均で正規化(T2S の出力も同じ正規化)
                raw = raw_targets([ref], [rc], f0)[0]
                rf = normalize(raw, mean_lf0(raw), stats[1])
            else:
                rf = f0_targets([ref], [rc], f0, stats)[0][0]
            if s_opt.get("no_prompt"):  # 参照音声を前につなげない
                rc, rf = rc[:0], rf[:0]
            rm = mel_norm.normalize(mels[ref.id].float())[: len(rc) * FPT].cuda()
            P = len(rc) * FPT
            # 参照のトークン・F0 を前につなげて、まとめて S2M に通す(詰め物は mask で除く)
            ids_all = torch.nn.utils.rnn.pad_sequence([torch.cat([rc, c.cpu()]) for c in codes], batch_first=True).cuda()
            f_all = torch.nn.utils.rnn.pad_sequence([torch.cat([rf, f.cpu()]) for f in fs], batch_first=True).cuda().float()
            h = codec.decode_hidden(ids_all) + f0cond(f_all)                                       # [B, Ur + U, Dc]
            lens = torch.tensor([(len(rc) + len(c)) * FPT for c in codes], device="cuda")
            T = int(lens.max())
            mask = torch.arange(T, device="cuda")[None] < lens[:, None]
            pm = torch.zeros(len(codes), T, rm.shape[-1], device="cuda")
            pm[:, :P] = rm
            style = spk[ref.id][None].expand(len(codes), -1).cuda()
            ms = s2m.generate(h, mask, pm, torch.full((len(codes),), P, device="cuda"),
                              torch.zeros_like(style) if s_opt.get("no_style") else style, **S2M_KW)
            for m, n in zip(ms, lens):
                waves.append(voc.generate(mel_norm.denormalize(m[P:n])[None])[0].cpu().numpy())
        return waves

    speakers = list(refs) if args.cond == "spk" else ["jsut"]
    if args.speakers and args.cond == "spk":
        speakers = [s for s in speakers if s in args.speakers]
    for seed in range(args.seeds):
        for s in speakers:
            for group, items in [("test", test_texts), ("practical", practical)]:
                seed_everything(seed)
                d = out_dir / f"seed{seed}" / group / s
                d.mkdir(parents=True, exist_ok=True)
                for (tid, _), w in zip(items, synth([t for _, t in items], refs.get(s))):
                    sf.write(d / f"{tid}.wav", w, SR)
        log(f"seed {seed} done ({len(speakers)} speakers)")
    log("done")


if __name__ == "__main__":
    main()
