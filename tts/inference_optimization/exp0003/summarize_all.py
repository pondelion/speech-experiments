"""量子化exp全体のseed 0・3回計測を集約し、最終散布図を作る。"""
import csv
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
OUT=RESULTS/'summary'


def read(path):return json.loads(path.read_text())


def collect():
    rows=[]
    dynamic_labels={'cpu_s2m_int8':'S2Mのみ動的INT8','cpu_t2s_int8':'T2Sのみ動的INT8','cpu_both_int8':'両部品動的INT8'}
    for index,name in enumerate(['cpu_s2m_int8','cpu_t2s_int8','cpu_both_int8'],1):
        speed=read(RESULTS/f'{name}.json')
        quality=read(RESULTS/('quality.json' if index==1 else 'quality_b.json'))
        rows.append(dict(id=f'B{index:02}',phase='B',condition=name,label=dynamic_labels[name],device='cpu',
            medium_s=speed['cases']['medium']['stages']['end_to_end_s']['median_s'],
            test_cer_pct=quality['groups']['test']['conditions'][name]['cer']['reading']*100,
            eval_texts_cer_pct=quality['groups']['eval_texts']['conditions'][name]['cer']['reading']*100,
            state_mib=(speed.get('t2s_state_bytes',216661159)+speed['s2m_state_bytes'])/2**20))
    c_labels=['FP32','重みのみINT8','FP32','BF16','重みのみINT8','GPTQ INT8',
              '単純INT4 g128','単純INT4 g64','単純INT4 g32','AWQ INT4','RTN INT4→Tinygemm','GPTQ INT4→Tinygemm']
    for index,row in enumerate(read(RESULTS/'weight_only/c_summary.json')['conditions'],1):
        rows.append(dict(id=f'C{index:02}',phase='C',condition=row['condition'],label=c_labels[index-1],device=row['device'],
            medium_s=row['medium_median_s'],test_cer_pct=row['test_reading_cer_percent'],
            eval_texts_cer_pct=row['practical_reading_cer_percent'],state_mib=row['t2s_s2m_state_mib']))
    for phase,folder in [('D','static'),('E','mixed'),('F','qat'),('G','runtime')]:
        for index,row in enumerate(read(RESULTS/folder/f'{phase.lower()}_summary.json'),1):
            if not row['complete']:continue
            rows.append({**{k:row[k] for k in ['condition','device','medium_s','test_cer_pct','eval_texts_cer_pct','state_mib']},
                         'id':f'{phase}{index:02}','phase':phase,'label':row['method']+(' / Linear compile' if phase=='D' else '')})
    return rows


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    rows=collect()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'all_conditions.json').write_text(json.dumps({'quality_seed':0,'speed_repeats':3,'latency_case':'medium',
        'note':'Different phases were measured at different times. Backend, dtype and quantized scope differ. No cross-phase causal speedup claim.',
        'conditions':rows},indent=2)+'\n')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=['id','phase','condition','label','device','medium_s','test_cer_pct','eval_texts_cer_pct','state_mib'])
    writer.writeheader();writer.writerows(rows);(OUT/'all_conditions.csv').write_text(buffer.getvalue())
    colors=dict(zip('BCDEFG',['#1f77b4','#e67e22','#2ca02c','#9467bd','#d62728','#009caa']))
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for i,device in enumerate(['cpu','cuda']):
        for j,(key,label) in enumerate([('test_cer_pct','Test: 100 texts'),('eval_texts_cer_pct','Practical: 55 texts')]):
            ax=axes[i,j];selected=[r for r in rows if r['device']==device]
            for n,row in enumerate(selected):
                is_control=any(s in row['condition'] for s in ['fp32','bf16','baseline']) or row['condition'].endswith('_torch')
                # ORT FP32 is a runtime candidate as well as a precision control.
                ax.scatter(row['medium_s'],row[key],c=colors[row['phase']],marker='s' if is_control else 'o',s=46,alpha=.85)
                offset=[(5,6),(5,-11),(-18,8),(-18,-12),(7,16),(-17,18)][n%6]
                ax.annotate(row['id'],(row['medium_s'],row[key]),xytext=offset,textcoords='offset points',fontsize=6,
                            arrowprops={'arrowstyle':'-','lw':.4,'color':'#999999'})
            ax.set(title=f'{device.upper()} / {label}',xlabel='Medium-text end-to-end latency (s), lower is better',ylabel='Reading CER (%), lower is better')
            ax.grid(alpha=.25);ax.margins(.18,.20)
    handles=[Line2D([],[],color=color,marker='o',ls='',label=phase) for phase,color in colors.items()]
    fig.legend(handles=handles,loc='outside upper center',ncol=6,title='Phase (point IDs map to the table / CSV)')
    fig.suptitle('TTS quantization: measured latency vs reading CER\nSeed 0; 3 timing repeats; phases have distinct controls',fontsize=13)
    fig.savefig(OUT/'speed_cer.svg',metadata={'Date':None})
    plt.close(fig)
    table=['| ID | 段階 | 環境 | 条件 | 中文 秒 | テストCER % | 実用文CER % | 重み MiB |',
           '|---|---|---|---|---:|---:|---:|---:|']
    for r in rows:table.append(f"| {r['id']} | {r['phase']} | {r['device']} | {r['label']} | {r['medium_s']:.3f} | {r['test_cer_pct']:.2f} | {r['eval_texts_cer_pct']:.2f} | {r['state_mib']:.1f} |")
    readme=ROOT/'README.md';text=readme.read_text();begin,end='<!-- ALL_RESULTS_START -->','<!-- ALL_RESULTS_END -->'
    block=begin+'\n'+'\n'.join(table)+'\n'+end
    if begin not in text:raise RuntimeError('READMEに最終集計マーカーが必要です')
    text=text[:text.index(begin)]+block+text[text.index(end)+len(end):];readme.write_text(text)
    print(f'Final scatter/table: {len(rows)} conditions')


if __name__=='__main__':main()
