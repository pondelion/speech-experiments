"""Eの事前定義した8条件を再開可能な形で実行する。"""
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
VARIANTS = [('cuda',mode) for mode in ['baseline','all','t2s','s2m','boundary']] + [
            ('cpu',mode) for mode in ['baseline','all','boundary']]


def main():
    if '--cpu-quality' in sys.argv:
        def quality(mode):
            subprocess.run([sys.executable,str(ROOT/'run_e.py'),'--device','cpu',
                            '--mode',mode,'--phase','quality'],check=True,cwd=ROOT)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(quality,['baseline','all','boundary']))
        return
    for phase in ['bench','quality','memory']:
        for device,mode in VARIANTS:
            print(f'START E {phase} {device} {mode}',flush=True)
            subprocess.run([sys.executable,str(ROOT/'run_e.py'),'--device',device,
                            '--mode',mode,'--phase',phase],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'summarize_e.py')],check=True,cwd=ROOT)
    print('E jobs complete',flush=True)


if __name__ == '__main__':
    main()
