"""付録の実ファイル容量・速度・CER・メモリをREADMEへ集約する。"""
import csv
import io
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'ternary'
LABELS={'baseline':'BF16保存基準','int4':'T2S INT4基準','weight_only':'重みのみ三値化',
        'direct':'直接三値化＋入力INT8','ft_ptq':'追加学習後三値化＋入力INT8','qat':'三値QAT＋入力INT8','kd':'三値QAT＋蒸留＋入力INT8'}


def main():
    rows=[];conditions=[]
    variants=[('cuda',m) for m in LABELS]+[('cpu',m) for m in ['baseline','kd']]
    for device,mode in variants:
        condition=f'{device}_ternary_{mode}';row={'device':device,'mode':mode,'condition':condition,'label':LABELS[mode]}
        phases={}
        for phase in ['bench','quality','memory']:
            path=OUT/f'{phase}_{condition}.json'
            if path.exists():phases[phase]=json.loads(path.read_text())
        if 'bench' in phases:
            b=phases['bench'];row['status']=b.get('status','measuring')
            row['file_mib']=b.get('artifact_bytes',b['quantization']['artifact_bytes'])/2**20
            if 'cases' in b:
                row['expanded_state_mib']=sum(b['state_bytes'].values())/2**20
                for name,c in b['cases'].items():row[f'{name}_s']=c['stages']['end_to_end_s']['median_s']
        if 'quality' in phases:
            q=phases['quality']
            for name,g in q['groups'].items():row[f'{name}_cer_pct']=g['cer']['reading']*100
            conditions.append({'id':condition,'label':f'{device.upper()} {LABELS[mode]}',
                'cer':{name:g['cer']['reading'] for name,g in q['groups'].items()}})
        if 'memory' in phases:
            m=phases['memory'];row['rss_mib']=m['process_peak_rss_bytes']/2**20
            if device=='cuda':row['vram_mib']=m['inference_peak_vram_bytes']/2**20
        row['complete']=len(phases)==3
        if row['complete']:row['status']='complete'
        rows.append(row)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    buffer=io.StringIO();fields=list(dict.fromkeys(k for row in rows for k in row));writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (OUT/'summary.csv').write_text(buffer.getvalue())
    table=['| 環境 | 条件 | 中文 秒 | テストCER % | 実用文CER % | 保存 MiB | peak RSS MiB | peak VRAM MiB | 状態 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rows:
        vals=[row['device'],row['label']]+[f'{row[k]:.2f}' if k in row else '—' for k in ['medium_s','test_cer_pct','eval_texts_cer_pct','file_mib','rss_mib','vram_mib']]
        vals.append({'complete':'完了','failed_preflight':'生成長異常で除外'}.get(row.get('status'),'実行中'))
        table.append('| '+' | '.join(vals)+' |')
    p=ROOT/'README.md';s=p.read_text();begin,end='<!-- TERNARY_RESULTS_START -->','<!-- TERNARY_RESULTS_END -->'
    assert begin in s and end in s
    a=s.index(begin);b=s.index(end)+len(end);s=s[:a]+begin+'\n'+'\n'.join(table)+'\n'+end+s[b:];p.write_text(s)
    manifest=ROOT.parents[2]/'site'/'manifest.json';text=manifest.read_text();data=json.loads(text)
    exp=next(e for b in data['bases'] if b['id']=='inference_optimization' for e in b['exps'] if e['id']=='exp0003')
    title='付録: 三値量子化・QAT・蒸留';groups=exp['condition_groups'];groups[:]=[g for g in groups if g['title']!=title]
    if conditions:
        groups.append({'title':title,'conditions':conditions});base=text.index('"id": "inference_optimization"');a=text.index('[',text.index('"condition_groups"',base));_,length=json.JSONDecoder().raw_decode(text[a:])
        replacement=json.dumps(groups,ensure_ascii=False,indent=2).replace('\n','\n          ');manifest.write_text(text[:a]+replacement+text[a+length:])
    print(f'Ternary: {sum(r["complete"] for r in rows)}/9 条件完了')


if __name__=='__main__':main()
