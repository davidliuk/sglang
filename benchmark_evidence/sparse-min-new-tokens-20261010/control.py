import importlib.util,pathlib,os,subprocess,signal,time,json,statistics,hashlib,requests
ROOT=pathlib.Path('/content/min-tokens');OUT=ROOT/'control';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('experiment',ROOT/'e2e-final.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
# Both representations/counters are maintained regardless of selection.
# The wrapper is therefore a diagnostic, not the original serving benchmark.
args=['--model-path',str(ROOT/'model'),'--attention-backend','triton','--sampling-backend','pytorch','--dtype','bfloat16','--host','127.0.0.1','--port','30000','--max-running-requests','128','--max-total-tokens','32768','--context-length','4096','--mem-fraction-static','0.7','--chunked-prefill-size','8192','--cuda-graph-bs-decode','1','8','32','128','--disable-prefill-cuda-graph','--stream-interval','1','--random-seed','42','--log-level','warning']
env={**os.environ,'PATH':'/content/sglang-env/bin:'+os.environ.get('PATH',''),'PYTHONPATH':'/content/sglang-src/python','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
log=open(OUT/'server.log','w');p=subprocess.Popen(['/content/sglang-env/bin/python',str(ROOT/'control-launcher.py'),*args],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
try:
    for _ in range(300):
        if p.poll() is not None:raise RuntimeError((OUT/'server.log').read_text()[-10000:])
        try:
            if requests.get(e.URL+'/health',timeout=1).status_code==200:break
        except requests.RequestException:pass
        time.sleep(1)
    else:raise RuntimeError('Control server startup timeout')
    rows=[]
    for variant in ['baseline','pr']:
        for name in ['single','batch128']:
            for item in e.FIXTURES[name]['sampling_params']:item['custom_params']={'bench_variant':variant}
            e.run_group(name)
    for cycle in range(4):
        for phase,variant in enumerate(['baseline','pr','pr','baseline']):
            for name in ['single','batch128']:
                for item in e.FIXTURES[name]['sampling_params']:item['custom_params']={'bench_variant':variant}
                row=e.run_group(name,stream=(name=='single'));row.update(cycle=cycle,phase=phase,variant=variant);rows.append(row)
            (OUT/'raw.json').write_text(json.dumps(rows,indent=2))
    for name in ['single','batch128']:
        group=[r for r in rows if r['case']==name];ref=group[0]['outputs']
        for row in group[1:]:
            assert all(a['output_ids']==b['output_ids'] and a['text']==b['text'] for a,b in zip(ref,row['outputs'])),(name,row['variant'],row['cycle'])
    (OUT/'DONE.json').write_text(json.dumps({'groups':len(rows),'responses':sum(len(r['outputs']) for r in rows),'exact_output_ids_and_text':True,'args':args,'baseline_sha256':hashlib.sha256((ROOT/'baseline_min_new_tokens.py').read_bytes()).hexdigest(),'pr_sha256':hashlib.sha256((ROOT/'pr_min_new_tokens.py').read_bytes()).hexdigest()},indent=2))
except BaseException as exc:(OUT/'FAILED.json').write_text(json.dumps({'error':repr(exc)}));raise
finally:e.stop(p,log)
