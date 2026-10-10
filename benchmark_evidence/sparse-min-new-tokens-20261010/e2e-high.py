import asyncio,hashlib,json,os,pathlib,signal,statistics,subprocess,sys,time
import aiohttp,requests
from transformers import AutoTokenizer
ROOT=pathlib.Path('/content/min-tokens/high');ROOT.mkdir(exist_ok=True); SOURCE=pathlib.Path('/content/sglang-src/python');URL='http://127.0.0.1:30000'
TARGET=SOURCE/'sglang/srt/sampling/penaltylib/min_new_tokens.py'
tok=AutoTokenizer.from_pretrained(pathlib.Path('/content/min-tokens/model'),local_files_only=True)
CASES=[('batch512',512,64),('control512',512,0)]
FIXTURES={}
for name,b,limit in CASES:
    inputs=[];params=[]
    for i in range(b):
        text=tok.apply_chat_template([{'role':'user','content':f'Entry {i}. Explain why trees need water. Include a numbered list of ten useful facts.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs.append(tok.encode(text,add_special_tokens=False))
        p={'temperature':0,'max_new_tokens':64,'min_new_tokens':([0,8,32,64][i%4] if limit=='mixed' else limit)}
        # Mixed sources, ragged stop lists, mixed penalties, natural stop boundaries.
        if limit=='mixed':
            p['stop_token_ids']=tok.encode(['apple','garden','soil','water'][i%4],add_special_tokens=False)
            p['frequency_penalty']=0.2 if i%3==0 else 0
            p['presence_penalty']=0.1 if i%5==0 else 0
        params.append(p)
    FIXTURES[name]={'input_ids':inputs,'sampling_params':params}
NATURAL=[]
for i,limit in enumerate([0,1,4,8]):
    text=tok.apply_chat_template([{'role':'user','content':f'Test {i}. Reply with just the numeral for 2+2. No explanation.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    NATURAL.append({'input_ids':tok.encode(text,add_special_tokens=False),'sampling_params':{'temperature':0,'min_new_tokens':limit,'max_new_tokens':64}})
(ROOT/'natural-fixtures.json').write_text(json.dumps(NATURAL,indent=2))
(ROOT/'fixtures.json').write_text(json.dumps(FIXTURES,indent=2))
def post(path,payload=None):
    r=requests.post(URL+path,json=payload,timeout=180);r.raise_for_status();return r.json()
def flush():
    r=requests.post(URL+'/flush_cache?timeout=30',timeout=35);r.raise_for_status()
def run_group(name,stream=False,cached=False):
    fixture=FIXTURES[name]
    if cached:post('/generate',{'input_ids':fixture['input_ids'],'sampling_params':{'temperature':0,'max_new_tokens':1}})
    else:flush()
    start=time.perf_counter()
    if stream:
        response=requests.post(URL+'/generate',json={**fixture,'stream':True},stream=True,timeout=180);response.raise_for_status()
        first={};outputs={}
        for line in response.iter_lines():
            if not line.startswith(b'data:'):continue
            raw=line[5:].strip()
            if raw==b'[DONE]':break
            out=json.loads(raw)
            if 'error' in out:raise RuntimeError(out)
            index=out.get('index',0);outputs[index]=out
            if out['meta_info']['completion_tokens']>0:first.setdefault(index,time.perf_counter()-start)
        result=[outputs[i] for i in range(len(fixture['input_ids']))]
        ttft=statistics.mean(first.values())*1000
    else:
        result=post('/generate',fixture);ttft=None
    elapsed=time.perf_counter()-start
    assert isinstance(result,list) and len(result)==len(fixture['input_ids']),(name,result)
    tokens=sum(o['meta_info']['completion_tokens'] for o in result)
    for o,p in zip(result,fixture['sampling_params']):
        assert len(o['output_ids'])==o['meta_info']['completion_tokens'],o
        assert o['meta_info']['completion_tokens']>=p['min_new_tokens'],(name,o,p)
        assert o['meta_info']['cached_tokens']>0 if cached else o['meta_info']['cached_tokens']==0,(name,o)
    return {'case':name,'cached':cached,'stream':stream,'wall_ms':elapsed*1000,'tokens_s':tokens/elapsed,'completion_tokens':tokens,'ttft_ms':ttft,'outputs':result}
def stop(p,log):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
    log.close();time.sleep(1)
def start(variant,mode,block):
    TARGET.write_bytes((ROOT/f'{variant}_min_new_tokens.py').read_bytes())
    label=f'{mode}-{block}-{variant}'
    args=['--model-path',str(pathlib.Path('/content/min-tokens/model')),'--attention-backend','triton','--sampling-backend','pytorch','--dtype','bfloat16','--host','127.0.0.1','--port','30000','--max-running-requests','512','--max-total-tokens','98304','--context-length','4096','--mem-fraction-static','0.7','--chunked-prefill-size','8192','--cuda-graph-bs-decode','1','8','32','128','512','--disable-prefill-cuda-graph','--disable-radix-cache','--stream-interval','1','--random-seed','42','--log-level','warning']
    if mode=='nonoverlap':args.append('--disable-overlap-schedule')
    env={**os.environ,'PATH':str(pathlib.Path(sys.executable).parent)+':'+os.environ.get('PATH',''),'PYTHONPATH':str(SOURCE),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    log=open(ROOT/f'{label}-server.log','w');p=subprocess.Popen([sys.executable,'-m','sglang.launch_server',*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    began=time.monotonic()
    try:
        for _ in range(300):
            if p.poll() is not None:
                stop(p,log);raise RuntimeError((ROOT/f'{label}-server.log').read_text()[-15000:])
            try:
                if requests.get(URL+'/health',timeout=1).status_code==200:
                    (ROOT/f'{label}-info.json').write_text(json.dumps({'ready_s':time.monotonic()-began,'args':args,'source_sha':subprocess.check_output(['git','-C','/content/sglang-src','rev-parse','HEAD'],text=True).strip(),'patch_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'kernel_sha256':hashlib.sha256((SOURCE/'sglang/kernels/ops/sampling/min_new_tokens.py').read_bytes()).hexdigest(),'server':requests.get(URL+'/get_server_info',timeout=15).json()},indent=2));print('READY',label,flush=True);return p,log
            except requests.RequestException:pass
            time.sleep(1)
        stop(p,log);raise RuntimeError('startup timeout')
    except BaseException:
        stop(p,log)
        raise

def main():
    rows=[];natural=[]
    for mode in ('overlap',):
        for block,variant in enumerate(('baseline','pr','pr','baseline')):
            p,log=start(variant,mode,block)
            try:
                for item in NATURAL:
                    flush()
                    result=post('/generate',item)
                    assert result['meta_info']['completion_tokens'] >= item['sampling_params']['min_new_tokens'],result
                    natural.append({'mode':mode,'block':block,'variant':variant,'request':item,'response':result})
                (ROOT/'natural-raw.json').write_text(json.dumps(natural,indent=2))
                for name,_,_ in CASES:run_group(name)
                # This extension measures only the 512-request workload.
                for sample in range(5):
                    for name,_,_ in CASES:
                        r=run_group(name);r.update(mode=mode,variant=variant,block=block,sample=sample);rows.append(r)
                    for name,cached in []:
                        r=run_group(name,stream=True,cached=cached);r.update(mode=mode,variant=variant,block=block,sample=sample);rows.append(r)
                    (ROOT/'e2e-raw.json').write_text(json.dumps(rows,indent=2))
                print('BLOCK_DONE',mode,block,variant,len(rows),flush=True)
            finally:stop(p,log)
    differences=[]
    for mode in ('overlap',):
        for name,_,_ in CASES:
            for cached,stream in ((False,False),(False,True),(True,True)):
                group=[r for r in rows if (r['mode'],r['case'],r['cached'],r['stream'])==(mode,name,cached,stream)]
                if not group:continue
                ref=group[0]['outputs']
                for r in group[1:]:
                    for i,(expected,actual) in enumerate(zip(ref,r['outputs'])):
                        if expected['output_ids']!=actual['output_ids'] or expected['text']!=actual['text']:
                            differences.append(dict(mode=mode,case=name,cached=cached,stream=stream,block=r['block'],variant=r['variant'],sample=r['sample'],row=i,expected=expected,actual=actual))
    for i in range(len(NATURAL)):
        rs=natural[i::len(NATURAL)];ref=rs[0]['response']
        for item in rs[1:]:
            out=item['response']
            assert out['output_ids']==ref['output_ids'] and out['text']==ref['text'],('natural',i,item)
    assert any(item['response']['meta_info']['finish_reason'].get('type')=='stop' for item in natural),'No natural stop was observed'
    (ROOT/'natural-parity.json').write_text(json.dumps({'responses':len(natural),'exact':True,'finish_reasons':[x['response']['meta_info']['finish_reason'] for x in natural]},indent=2))
    (ROOT/'e2e-parity.json').write_text(json.dumps({'groups':len(rows),'responses':sum(len(r['outputs']) for r in rows),'differences':differences},indent=2))
    print('PARITY',len(differences),'differences',flush=True)
    assert not differences,'Output differences: inspect full raw records'
if __name__=='__main__':main()
