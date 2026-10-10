import hashlib,importlib.util,json,pathlib,statistics,time,types,torch,random
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
ROOT=pathlib.Path('/content/min-tokens')
class Batch(types.SimpleNamespace):
    pass
def load(variant):
    spec=importlib.util.spec_from_file_location(variant,ROOT/f'{variant}_min_new_tokens.py'); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.BatchedMinNewTokensPenalizer
def make(cls,b,v,k,device='cuda'):
    reqs=[]
    for i in range(b):
        reqs.append(types.SimpleNamespace(sampling_params=types.SimpleNamespace(min_new_tokens=16+i%4,stop_token_ids=set(range(k))),eos_token_ids={v-1},tokenizer=types.SimpleNamespace(eos_token_id=v-1,additional_stop_token_ids=None)))
    batch=Batch(reqs=reqs,device=device)
    orch=BatchedPenalizerOrchestrator(v,batch,{cls}); return batch,orch,orch.penalizers[cls]
classes={v:load(v) for v in ('baseline','pr')}; records=[]
# Exact same real production class and orchestrator; no operator emulation.
for b in (1,8,32,128,256,512):
    for v in (32000,151936):
        for k in (1,8):
            pens={n:make(c,b,v,k) for n,c in classes.items()}
            logits=torch.randn(b,v,device='cuda'); outputs={}
            for n,(_,_,p) in pens.items():
                x=logits.clone();p.apply(x);outputs[n]=x
            torch.testing.assert_close(outputs['baseline'],outputs['pr'],rtol=0,atol=0)
            for n,(_,_,p) in pens.items():
                for _ in range(20):p.apply(logits)
            torch.cuda.synchronize()
            for block,n in enumerate(('baseline','pr','pr','baseline')):
                p=pens[n][2]
                for sample in range(10):
                    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    begin.record();t=time.perf_counter()
                    for _ in range(100):p.apply(logits)
                    end.record();end.synchronize()
                    records.append(dict(batch=b,vocab=v,stops=k,variant=n,block=block,sample=sample,gpu_us=begin.elapsed_time(end)*10,wall_us=(time.perf_counter()-t)*1e4))
            (ROOT/'micro-raw.json').write_text(json.dumps(records,indent=2))
# CUDA graph replay respects the limit boundary and empty/ragged stop lists.
_,_,p=make(classes['pr'],4,32000,3)
x=torch.zeros(4,32000,device='cuda');s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):p.apply(x)
torch.cuda.current_stream().wait_stream(s)
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):p.apply(x)
for count in (0,16,17,18,19,20):
    p.len_output_tokens.fill_(count);x.fill_(0);g.replay();torch.cuda.synchronize()
    for row in range(4):
        assert bool(torch.isneginf(x[row,0]))==(count<16+row)
        assert x[row,4].item()==0
(ROOT/'micro-checks.json').write_text(json.dumps({'exact_shapes':24,'cuda_graph_boundary_counts':[0,16,17,18,19,20],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'baseline_sha256':hashlib.sha256((ROOT/'baseline_min_new_tokens.py').read_bytes()).hexdigest(),'pr_sha256':hashlib.sha256((ROOT/'pr_min_new_tokens.py').read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256(pathlib.Path('/content/sglang-src/python/sglang/kernels/ops/sampling/min_new_tokens.py').read_bytes()).hexdigest()},indent=2))
print('MICRO_PASS',len(records),flush=True)
