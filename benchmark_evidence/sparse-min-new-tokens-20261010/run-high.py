import pathlib,subprocess,os,json,time,shutil
r=pathlib.Path('/content/min-tokens');high=r/'high';high.mkdir(exist_ok=True)
for n in ['baseline_min_new_tokens.py','pr_min_new_tokens.py']:(high/n).write_bytes((r/n).read_bytes())
env={**os.environ,'PATH':'/content/sglang-env/bin:'+os.environ.get('PATH',''),'PYTHONPATH':'/content/sglang-src/python','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
timings={}
try:
    for name,path in [('e2e','/content/min-tokens/e2e-high.py')]:
        t=time.monotonic()
        with open(high/(name+'.log'),'w') as log:subprocess.run(['/content/sglang-env/bin/python',path],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        timings[name]=time.monotonic()-t;print('DONE',name,timings[name],flush=True)
    pathlib.Path('/content/sglang-src/python/sglang/srt/sampling/penaltylib/min_new_tokens.py').write_bytes((r/'pr_min_new_tokens.py').read_bytes())
    (high/'DONE.json').write_text(json.dumps(timings,indent=2))
except BaseException as e:(high/'FAILED.json').write_text(json.dumps({'error':repr(e),'timings':timings}));raise
