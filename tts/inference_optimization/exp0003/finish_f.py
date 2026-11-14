"""Fの固定5条件を再開可能な形で実行する。"""
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parent
VARIANTS = [('cuda',mode) for mode in ['baseline','ptq','ft','ft_ptq','qat']]


def main():
    for phase in ['bench','quality','memory']:
        for device,mode in VARIANTS:
            print(f'START F {phase} {device} {mode}',flush=True)
            subprocess.run([sys.executable,str(ROOT/'run_f.py'),'--device',device,
                            '--mode',mode,'--phase',phase],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'summarize_f.py')],check=True,cwd=ROOT)
    print('F jobs complete',flush=True)


if __name__ == '__main__':
    main()
