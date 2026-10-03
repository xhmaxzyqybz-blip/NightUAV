"""Bounded loader microbenchmark. No training; original train files read-only."""
import argparse, hashlib, json, statistics, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'experiments/exp004a_multisample_smoke/smoke_train_manifest.json').read_text(encoding='utf-8'))
    paths=[root/r['source'] for r in manifest['rows'][:3]]
    assert torch.cuda.is_available(), 'CUDA required'
    before={str(p.relative_to(root)):sha(p) for p in paths}
    torch.zeros(1,device='cuda');torch.cuda.synchronize()
    records=[];cache=[]
    def tensor(a):return torch.from_numpy(a/255.0).float().permute(2,0,1)
    for repeat in range(3):
        for p in paths:
            row={'source':p.relative_to(root).as_posix(),'repeat':repeat}
            t=time.perf_counter()
            with Image.open(p) as im: rgb=im.convert('RGB')
            row['decode_rgb_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();arr=np.array(rgb);row['numpy_copy_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();blur=cv2.GaussianBlur(arr,(9,9),sigmaX=1.7,sigmaY=1.7);row['blur_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();g=tensor(blur);x=tensor(np.asarray(rgb));row['full_normalize_tensor_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter();xc=x[:,32:288,48:304].unsqueeze(0).contiguous();gc=g[:,32:288,48:304].unsqueeze(0).contiguous();row['crop_contiguous_ms']=(time.perf_counter()-t)*1000
            torch.cuda.synchronize();t=time.perf_counter();xd=xc.cuda();gd=gc.cuda();torch.cuda.synchronize();row['h2d_ms']=(time.perf_counter()-t)*1000
            row['total_components_ms']=sum(v for k,v in row.items() if k.endswith('_ms'))
            t=time.perf_counter();cx=tensor(arr[32:288,48:304]).unsqueeze(0).contiguous();cg=tensor(blur[32:288,48:304]).unsqueeze(0).contiguous();row['cached_uint8_crop_convert_ms']=(time.perf_counter()-t)*1000
            row['bitwise_equal']=torch.equal(xc,cx) and torch.equal(gc,cg)
            assert row['bitwise_equal']
            row['size']=list(rgb.size)
            if repeat==0:cache.append((arr,blur))
            records.append(row)
            del x,g,xd,gd
    assert before=={str(p.relative_to(root)):sha(p) for p in paths}
    report={'scope':'3 original train images x 3 repeats; warm OS cache possible; no disk cold-cache claim; no training; fixed crop; no DataLoader workers benchmarked',
            'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'torch_threads':torch.get_num_threads(),'opencv_threads':cv2.getNumThreads(),
            'source_hashes':before,'records':records,'all_crops_bitwise_equal':True,'inputs_unchanged':True,
            'cache_bytes_3_uint8_pairs':sum(a.nbytes+b.nbytes for a,b in cache),
            'mean_ms':{k:statistics.mean(r[k] for r in records) for k in records[0] if k.endswith('_ms')}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['mean_ms'],indent=2))
if __name__=='__main__':main()
