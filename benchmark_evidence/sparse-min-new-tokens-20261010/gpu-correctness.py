import pathlib,json,torch,types,importlib.util,random
from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
ROOT=pathlib.Path('/content/min-tokens')
class Batch(types.SimpleNamespace):
    pass
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'{name}_min_new_tokens.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.BatchedMinNewTokensPenalizer
classes={name:load(name) for name in ['baseline','pr']}
def make(cls,reqs,v):
    batch=Batch(reqs=reqs,device='cuda');orch=BatchedPenalizerOrchestrator(v,batch,{cls});return batch,orch,orch.penalizers[cls]
def req(limit,ids,eos):return types.SimpleNamespace(sampling_params=types.SimpleNamespace(min_new_tokens=limit,stop_token_ids=set(ids)),eos_token_ids=set(eos),tokenizer=types.SimpleNamespace(eos_token_id=None,additional_stop_token_ids=set(ids[::2])))
random.seed(42);torch.manual_seed(42);checks=0
for dtype in [torch.float32,torch.float16,torch.bfloat16]:
    for b in [1,7,32]:
        v=257
        rs=[req(i%4,random.sample(range(v),i%11),[v-1]) for i in range(b)]
        pens={n:make(c,rs,v) for n,c in classes.items()}
        for step in range(6):
            logits=torch.randn(b,v*2,device='cuda',dtype=dtype)[:,::2];out={}
            for n,(_,orch,p) in pens.items():
                x=torch.empty_strided(logits.shape,logits.stride(),device='cuda',dtype=dtype);x.copy_(logits);orch.apply(x);out[n]=x;p.cumulate_output_tokens(torch.zeros(b,device='cuda',dtype=torch.int64))
            torch.testing.assert_close(out['baseline'],out['pr'],rtol=0,atol=0);checks+=1
        for n,(_,orch,p) in pens.items():p.filter(torch.arange(0,b,2,device='cuda'));other=make(classes[n],[req(3,[0,4,8,11],[])],v);p.merge(other[2]);pens[n]=(*pens[n][:2],p)
        logits=torch.randn((b+1)//2+1,v,device='cuda',dtype=dtype);out={}
        for n,(_,orch,p) in pens.items():x=logits.clone();p.apply(x);out[n]=x
        torch.testing.assert_close(out['baseline'],out['pr'],rtol=0,atol=0);checks+=1
# Empty stop lists, and both compact->wide and wide->compact merges.
for wide_first in [False,True]:
    rs=[req(2,[],[])] if not wide_first else [req(2,[0,5,7],[256])]
    other_rs=[req(3,[1,2,3],[256])] if not wide_first else [req(3,[],[])]
    out={}
    for n,c in classes.items():
        _,orch,p=make(c,rs,257);other=make(c,other_rs,257);p.merge(other[2]);x=torch.ones(2,257,device='cuda');p.apply(x);out[n]=x
    torch.testing.assert_close(out['baseline'],out['pr'],rtol=0,atol=0);checks+=1
(ROOT/'gpu-correctness.json').write_text(json.dumps({'exact_checks':checks,'dtypes':['float32','float16','bfloat16'],'strided_logits':True,'filter_merge':True},indent=2));print('GPU_CORRECTNESS_PASS',checks)
