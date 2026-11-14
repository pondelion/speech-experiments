"""Fの完了済み測定を集約する。未完了の測定は空欄で保持する。"""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'qat'


def main():
    rows = []
    conditions = []
    variants = [('cuda',m,l) for m,l in [('baseline','BF16基準'),('ptq','T2S PTQ INT4'),('ft','追加学習 BF16'),('ft_ptq','追加学習後PTQ INT4'),('qat','QAT後INT4')]]
    for device,mode,label in variants:
        condition = f'{device}_f_{mode}'
        row = dict(condition=condition, device=device, method=label)
        phases = {}
        for phase in ['bench', 'quality', 'memory']:
            path = OUT / f'{phase}_{condition}.json'
            if path.exists():
                phases[phase] = json.loads(path.read_text())
        row['complete'] = len(phases) == 3
        if 'bench' in phases:
            b = phases['bench']
            row['state_mib'] = sum(b['state_bytes'].values()) / 2**20
            for name, case in b['cases'].items():
                row[f'{name}_s'] = case['stages']['end_to_end_s']['median_s']
            row['first_warmup_s'] = b['cases']['short']['warmup_s']
        if 'quality' in phases:
            q = phases['quality']
            for name, group in q['groups'].items():
                row[f'{name}_cer_pct'] = group['cer']['reading'] * 100
            conditions.append({'id': condition, 'label': f'{device.upper()} {label}',
                'cer': {n: g['cer']['reading'] for n, g in q['groups'].items()}})
        if 'memory' in phases:
            m = phases['memory']
            row['rss_mib'] = m['process_peak_rss_bytes'] / 2**20
            if device == 'cuda':
                row['vram_mib'] = m['inference_peak_vram_bytes'] / 2**20
        rows.append(row)
    (OUT / 'f_summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader(); writer.writerows(rows)
    (OUT / 'f_summary.csv').write_text(buf.getvalue())
    table = ['| Device | 条件 | 中文 秒 | テストCER % | 実用文CER % | 重み MiB | peak RSS MiB | peak VRAM MiB |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row['device'], row['method']]
        for key in ['medium_s','test_cer_pct','eval_texts_cer_pct','state_mib','rss_mib','vram_mib']:
            values.append(f'{row[key]:.2f}' if key in row else '—')
        table.append('| '+' | '.join(values)+' |')
    readme = ROOT / 'README.md'
    text = readme.read_text()
    begin, end = '<!-- F_RESULTS_START -->', '<!-- F_RESULTS_END -->'
    block = begin+'\n'+'\n'.join(table)+'\n'+end
    if begin in text:
        text = text[:text.index(begin)] + block + text[text.index(end)+len(end):]
    else:
        raise RuntimeError('READMEのF節に結果表の挿入マーカーが必要です')
    readme.write_text(text)
    manifest = ROOT.parents[2] / 'site' / 'manifest.json'
    manifest_text = manifest.read_text()
    data = json.loads(manifest_text)
    exp = next(e for b in data['bases'] if b['id']=='inference_optimization'
               for e in b['exps'] if e['id']=='exp0003')
    title = 'F: QATと追加学習の比較'
    groups = exp['condition_groups']
    groups[:] = [g for g in groups if g['title'] != title]
    if conditions:
        groups.append({'title':title, 'conditions':conditions})
        exp_start = manifest_text.index('"id": "inference_optimization"')
        start = manifest_text.index('[', manifest_text.index('"condition_groups"', exp_start))
        _, length = json.JSONDecoder().raw_decode(manifest_text[start:])
        replacement = json.dumps(groups,ensure_ascii=False,indent=2).replace('\n','\n          ')
        manifest.write_text(manifest_text[:start]+replacement+manifest_text[start+length:])
    print(f'F: {sum(r["complete"] for r in rows)}/5 条件で速度・CER・メモリ完了')


if __name__ == '__main__':
    main()
