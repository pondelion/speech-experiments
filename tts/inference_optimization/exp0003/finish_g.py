"""Gの7条件を既存結果から再開して比較する。"""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent
VARIANTS=[('cuda',m) for m in ['torch','fp32','int4']]+[('cpu',m) for m in ['torch','fp32','int8','int4']]

if __name__=='__main__':
    if '--prefetch-ort-quality' in sys.argv or '--prefetch-quant-quality' in sys.argv:
        for mode in (['int8','int4'] if '--prefetch-quant-quality' in sys.argv else ['fp32','int8','int4']):
            subprocess.run([sys.executable,str(ROOT/'run_g.py'),'--device','cpu','--mode',mode,'--phase','quality'],check=True,cwd=ROOT)
        sys.exit(0)
    for phase in ['bench','quality','memory']:
        for device,mode in VARIANTS:
            print('START G',phase,device,mode,flush=True)
            subprocess.run([sys.executable,str(ROOT/'run_g.py'),'--device',device,'--mode',mode,'--phase',phase],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'summarize_g.py')],check=True,cwd=ROOT)
    print('G jobs complete',flush=True)
