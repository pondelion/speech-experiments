"""同じマスク・時刻・条件を使う通常lossと出力分布の蒸留。"""

import torch
import torch.nn.functional as functional


def loss_for(student, teacher, data, indices, kd_weight=0.0):
    """固定encoderと長さ条件のもとで生成モデルだけを学習する。"""
    from exp0002.common.t2s_masked import mask_ratio

    texts = [data["texts"][i] for i in indices]
    codes = [data["codes"][i] for i in indices]
    durations = [data["durations"][i] for i in indices]
    time = torch.rand(len(codes)).clamp(1e-5, 1.0)
    masks = []
    for code, ratio in zip(codes, mask_ratio(time).clamp(min=0.2)):
        mask = torch.rand(len(code)) < ratio
        if not mask.any():
            mask[torch.randint(len(code), ())] = True
        masks.append(mask)
    masked = [
        torch.where(mask, student.n_codes, code)
        for code, mask in zip(codes, masks)
    ]
    drop = torch.rand(len(codes)) < student.text_drop
    with torch.no_grad():
        encoded, _ = teacher.encode(texts)
        condition = teacher.align(encoded, durations)
    selection = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True).to(
        student.device
    )
    labels = torch.cat([code[mask] for code, mask in zip(codes, masks)]).to(
        student.device
    )
    hidden, _ = student.hidden(texts, masked, time, drop, condition)
    logits = student.head(hidden[selection]).float()
    ce = functional.cross_entropy(logits, labels)
    kd = ce.new_zeros(())
    if kd_weight:
        with torch.no_grad():
            hidden_teacher, _ = teacher.hidden(
                texts, masked, time, drop, condition
            )
            teacher_logits = teacher.head(hidden_teacher[selection]).float()
        kd = functional.kl_div(
            logits.log_softmax(-1),
            teacher_logits.softmax(-1),
            reduction="batchmean",
        )
    loss = ce + kd_weight * kd
    return loss, {
        "ce": float(ce.detach()),
        "kl": float(kd.detach()),
        "accuracy": float((logits.argmax(-1) == labels).float().mean()),
        "masked_positions": len(labels),
    }
