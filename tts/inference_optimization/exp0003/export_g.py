"""GのFP32 exportとINT8/INT4への変換。重みはGit管理外に保存する。"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'exp0001'))
from benchmark import build_models
from onnx_models import export_models,quantized_paths

if __name__=='__main__':
    torch,pipe,t2s,*_=build_models('cpu')
    torch.set_num_threads(8)
    export_models(torch,pipe,t2s)
    for mode in ['int8','int4']:
        for path in quantized_paths(mode):print('QUANTIZED',path.name,path.stat().st_size,flush=True)
