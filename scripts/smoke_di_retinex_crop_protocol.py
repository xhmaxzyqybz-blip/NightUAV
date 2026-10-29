"""Only 200 updates; independently randomized strict-zero crop protocol smoke test."""
import json,random,time,hashlib,traceback,statistics
from pathlib import Path
import numpy as np
from PIL import Image
import train_di_retinex_exp005a as base
from nightuav_crop_adapter import select_crop,BlackCropRetryError
ROOT=base.ROOT;EXP=ROOT/'experiments/exp005a_crop_protocol_smoke'
def dump(name,v):base.dump(EXP/name,v)
def stats(records):
    bad=[x for x in records if x['invalid_crops']]
    return {'invalid_crop_resample_count':sum(len(x['invalid_crops']) for x in records),'affected_image_count':len({x['source'] for x in bad}),'affected_image_paths':sorted({x['source'] for x in bad}),'affected_sample_count':len(bad),'max_sample_resample_count':max((x['resample_count'] for x in records),default=0)}
def main():
    assert not EXP.exists(),'Refuse overwrite'
    torch,official,split,meta=base.setup();EXP.mkdir()
    protected=[ROOT/'scripts'/n for n in ['train_di_retinex_exp005a.py','continue_di_retinex_exp005a_15ep.py']]+[base.REPO/n for n in ['mymodel.py','lowlight_train.py','dataloader.py']]+list((ROOT/'configs/splits').glob('*v2_1*'))
    hashes={p:base.sha(p) for p in protected}
    config={'steps':200,'seed':1149,'random_initialization':True,'checkpoint_loaded':False,'batch':1,'crop':[256,256],'optimizer':'Adam','lr':.001,'weight_decay':.0001,'gradient_clip':.1,'RD_VS':[1,1],'exposure':.5,'max_retries':50,'retry_definition':'Initial crop plus at most 50 same-image replacement draws. If all 50 replacements are zero, raise with path.','reject_only':'all RGB channels at all crop pixels exactly zero','rng':'Separate random.Random(1149) streams for shuffle, initial crops, replacement crops. Main process only; num_workers=0.','split_counts':{k:len(v) for k,v in split.items()},'model_commit':base.COMMIT,'protected_sha256':{p.relative_to(ROOT).as_posix():h for p,h in hashes.items()},'script_sha256':base.sha(Path(__file__)),'adapter_sha256':base.sha(ROOT/'scripts/nightuav_crop_adapter.py'),'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__)}
    dump('config/run.json',config)
    # Controlled synthetic boundary and exact-zero tests.
    for value in [np.float32(1/255),np.float32(1e-12)]:
        a=np.zeros((3,256,256),np.float32);a[0,123,45]=value
        _,_,r=select_crop(a,'synthetic_nonzero',random.Random(1149));assert not r['invalid_crops'] and r['resample_count']==0
    zero=np.zeros((3,256,256),np.uint8)
    try:select_crop(zero,'synthetic_all_black',random.Random(1149));raise AssertionError('No exhaustion error')
    except BlackCropRetryError as ex:
        assert ex.record['resample_count']==50 and len(ex.record['invalid_crops'])==51 and 'synthetic_all_black' in str(ex);exhaustion=ex.record
    class LastValid:
        def __init__(self):self.calls=0
        def randint(self,a,b):
            self.calls+=1;return 1 if self.calls==99 else 0
    edge=np.zeros((3,256,257),np.uint8);edge[0,0,256]=1
    _,_,boundary=select_crop(edge,'synthetic_last_retry',LastValid(),(0,0));assert boundary['resample_count']==50 and len(boundary['invalid_crops'])==50
    unit_records=[];unit_summary=[]
    for name in ['0323.jpg','0651.jpg','0455.jpg','0069.jpg']:
        source='datasets/U3D/lowlight/train/'+name;assert source in split['train'];p=ROOT/source;before=base.sha(p)
        with Image.open(p) as im:a=np.array(im.convert('RGB')).transpose(2,0,1)
        array_hash=hashlib.sha256(a.tobytes()).hexdigest();records=[];rng=random.Random(1149)
        for trial in range(1000):
            x,y,r=select_crop(a,source,rng);assert np.any(a[:,y:y+256,x:x+256]!=0);r['trial']=trial+1;records.append(r)
        assert any(r['invalid_crops'] for r in records)
        # Replay identical draws without another decode.
        rng=random.Random(1149)
        for trial,expected in enumerate(records,1):
            _,_,r=select_crop(a,source,rng);r['trial']=trial;assert r==expected
        assert base.sha(p)==before==meta['image_sha256'][source] and hashlib.sha256(a.tobytes()).hexdigest()==array_hash
        unit_records+=records;unit_summary.append({'source':source,'samples':1000,'returned_nonzero':1000,'deterministic_replay':True,**stats(records)})
    # Test known failed crop with the actual official preprocessing and adapter.
    ds=base.PatchDataset(split['train']);idx=split['train'].index('datasets/U3D/lowlight/train/0069.jpg');ds[(idx,0,0)]
    X,G,path=ds.original[idx];assert not X[:,1850:2106,68:324].any()
    x,y,integration=select_crop(X.numpy(),split['train'][idx],random.Random(1149),(68,1850));assert X[:,y:y+256,x:x+256].any();assert integration['resample_count']>0
    dump('metrics/unit_tests.json',{'status':'PASS','dark_nonzero_accepted':True,'zero_exhaustion':exhaustion,'last_retry_accepted':boundary,'real_loader_forced_0069':integration,'high_risk':unit_summary,'aggregate':stats(unit_records),'source_arrays_and_files_unchanged':True})
    dump('metrics/sampler_trials.json',unit_records)
    print('UNIT TESTS PASS',json.dumps(unit_summary),flush=True)
    del X,G
    # New model, official initialization/loss; no checkpoint loading.
    official.set_seed(1149);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=official.mymodel.enhance_net_nopool().cuda();model.apply(official.weights_init);model.train();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001);loss_fn,expressions=base.make_loss(torch);dump('config/loss_expressions.json',expressions)
    order=list(range(770));random.Random(1149).shuffle(order);crops=random.Random(1149);retry=random.Random(1149);plan=[(i,crops.randint(0,3584),crops.randint(0,1904)) for i in order]
    rows=[];crop_records=[];start=time.perf_counter()
    with (EXP/'metrics/train_steps.jsonl').open('x',encoding='utf-8') as log:
        for step,(idx,x,y) in enumerate(plan[:200],1):
            t=time.perf_counter();X,G,path=ds.original[idx];source=split['train'][idx];assert Path(path).resolve()==(ROOT/source).resolve() and source not in split['dev']+split['quarantine']
            x,y,record=select_crop(X.numpy(),source,retry,(x,y));record['step']=step;crop_records.append(record)
            X=X[:,y:y+256,x:x+256].contiguous().unsqueeze(0);G=G[:,y:y+256,x:x+256].contiguous().unsqueeze(0);assert X.any();wait_ms=(time.perf_counter()-t)*1000
            torch.cuda.synchronize();t=time.perf_counter();X=X.cuda();G=G.cuda();torch.cuda.synchronize();transfer_ms=(time.perf_counter()-t)*1000
            before=[p.detach().clone() for p in model.parameters()];torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
            Y,c,b,k=model(X);rd,vs,loss=loss_fn(X,G,Y,b,k,.5);assert all(torch.isfinite(z).all().item() for z in [X,G,Y,c,b,k,rd,vs,loss]);opt.zero_grad();loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());grad=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();compute_ms=(time.perf_counter()-t)*1000
            delta=sum(float((p.detach()-q).abs().sum()) for p,q in zip(model.parameters(),before));assert delta>0 and all(torch.isfinite(p).all() for p in model.parameters())
            row={'step':step,'source':source,'crop':record,'total':float(loss.detach()),'RD':float(rd.detach()),'VS':float(vs.detach()),'gradient_finite':True,'gradient_norm':float(grad),'parameter_update_l1':delta,'cuda':X.is_cuda,'wait_ms':wait_ms,'transfer_ms':transfer_ms,'compute_ms':compute_ms,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20};rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();del X,G,Y,c,b,k,rd,vs,loss,before
            if step%25==0:print('TRAIN',step,'PASS',flush=True)
    seconds=time.perf_counter()-start
    assert len(rows)==200 and len({r['source'] for r in rows})==200
    assert all(base.sha(p)==h for p,h in hashes.items())
    assert all(base.sha(ROOT/p)==h for p,h in meta['image_sha256'].items())
    summary={'status':'PASS','steps':200,'unique_train_images':200,'training_crop_stats':stats(crop_records),'unit_crop_stats':stats(unit_records),'seconds':seconds,'mean_data_ms':statistics.mean(r['wait_ms'] for r in rows),'mean_compute_ms':statistics.mean(r['compute_ms'] for r in rows),'peak_allocated_mib':max(r['peak_allocated_mib'] for r in rows),'peak_reserved_mib':max(r['peak_reserved_mib'] for r in rows),'all_gradients_finite':True,'all_parameters_updated':True,'nan_inf':False,'oom':False,'original_network_loss_files_unchanged':True,'all_image_hashes_unchanged':True,'splits_unchanged':True,'no_dev_or_quarantine_training':True}
    dump('metrics/summary.json',summary);dump('metrics/train_history.json',rows)
    text='# EXP005A strict-zero crop protocol smoke\n\nPASS: 200 steps only, fresh random official initialization. No previous checkpoint loaded. No dev evaluation or model selection. No checkpoint saved or formal 15-epoch training started.\n\n## Locked protocol\n\n256x256 paired crops AFTER original full-image Gaussian processing. Reject ONLY if every RGB channel is exactly zero; draw replacements from the SAME image using controlled RNG. Initial attempt plus at most 50 replacement attempts. A valid 50th replacement is accepted; 50 invalid replacements raise BlackCropRetryError with source path and all coordinates. No epsilon, brightness threshold, alternate image or skipped sample. Gaussian crop uses the same accepted coordinates.\n\nThe adapter is scripts/nightuav_crop_adapter.py. Initial-coordinate, image-order and replacement RNGs are separate Random(1149) streams. Single-process loading avoids worker-dependent RNG. Old training scripts are preserved; future baseline must explicitly integrate this adapter and record the protocol version.\n\n## Training\n\nOriginal network and AST-extracted official RD+VS 1:1; Adam lr0.001, weight_decay0.0001, clip0.1, exposure0.5, batch1, seed1149. First 200 images of a shuffled permutation of train770; dev8/quarantine22 unchanged and unused. GPU, sources, hashes, commands and definitions are in config/run.json. Per-step loss, gradient, update and timing: metrics/train_steps.jsonl.\n\n## Tests\n\nEach high-risk image received 1000 independent requested crops with same-image retries, plus an exact deterministic replay. All returned crops were nonzero. Synthetic patches with a single 1/255 or 1e-12 channel value were accepted immediately. An all-zero image exhausted exactly 50 replacements and raised with its path; a controlled sequence succeeded on replacement50. The previously failing 0069 crop also passed through the actual official preprocessing and replacement adapter. All source hashes and original network/loss files remained unchanged.\n\n'
    text+='Training statistics:\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nHigh-risk sampler results:\n\n```json\n'+json.dumps(unit_summary,indent=2)+'\n```\n\nEvery invalid coordinate and retry number is retained in metrics/sampler_trials.json and train history. Unit-test counts and training counts are separate; synthetic exhaustion/forced integration tests are not added to stochastic rates. This proves technical smoke compatibility, not convergence, quality or generalization. No commit/push. Stop for human review.\n'
    with (EXP/'README.md').open('x',encoding='utf-8') as f:f.write(text)
    print('SMOKE COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        if EXP.exists():dump('failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'retry_record':getattr(e,'record',None)})
        raise