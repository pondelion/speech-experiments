"""Dの固定行列を逐次実行。既存結果は再実行しない。"""
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent


def main():
    for phase in ['bench','quality','memory']:
        for device in ['cuda','cpu']:
            for mode in ['fp32','static','sq05','sq075']:
                if device == 'cpu' and mode != 'fp32':
                    print(f'SCREENED OUT D {device} {mode}: uncompetitive CPU INT8 kernel; CER not evaluated', flush=True)
                    continue
                print(f'START D {phase} {device} {mode}',flush=True)
                subprocess.run([sys.executable,str(ROOT/'run_d.py'),'--device',device,
                    '--mode',mode,'--phase',phase],check=True,cwd=ROOT)
    print('D jobs complete',flush=True)


if __name__=='__main__':
    main()
