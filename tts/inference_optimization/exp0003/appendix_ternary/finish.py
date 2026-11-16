"""付録の学習とCPU/GPU比較を固定条件で実行する。"""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
VARIANTS=[('cuda',m) for m in ['baseline','int4','weight_only','direct','ft_ptq','qat','kd']]+[('cpu',m) for m in ['baseline','kd']]


def main():
    if '--cpu-quality' in sys.argv:
        from concurrent.futures import ThreadPoolExecutor
        def quality(mode):
            subprocess.run([sys.executable,str(HERE/'run.py'),'--device','cpu','--mode',mode,'--phase','quality'],check=True,cwd=ROOT)
        with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(quality,['baseline','kd']))
        return
    for mode in ['ft','qat','kd']:
        subprocess.run([sys.executable,str(HERE/'train.py'),'--mode',mode],check=True,cwd=ROOT)
    for phase in ['bench','quality','memory']:
        for device,mode in VARIANTS:
            bench=ROOT/'results'/'ternary'/f'bench_{device}_ternary_{mode}.json'
            if phase!='bench' and bench.exists() and json.loads(bench.read_text()).get('status')=='failed_preflight':continue
            print('START ternary',phase,device,mode,flush=True)
            subprocess.run([sys.executable,str(HERE/'run.py'),'--device',device,'--mode',mode,'--phase',phase],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True,cwd=ROOT)


if __name__=='__main__':main()
