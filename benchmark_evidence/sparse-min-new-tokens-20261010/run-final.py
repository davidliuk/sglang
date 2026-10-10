import pathlib,subprocess,os,json,time,hashlib
root=pathlib.Path('/content/min-tokens');final=root/'final';final.mkdir(exist_ok=True)
for n in ['baseline_min_new_tokens.py','pr_min_new_tokens.py']:(final/n).write_bytes((root/n).read_bytes())
python='/content/sglang-env/bin/python';env={**os.environ,'PATH':'/content/sglang-env/bin:'+os.environ.get('PATH',''),'PYTHONPATH':'/content/sglang-src/python','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
timings={}
def run(name,path):
    t=time.monotonic()
    with open(final/(name+'.log'),'w') as log:subprocess.run([python,str(root/path)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    timings[name]=time.monotonic()-t;print('DONE',name,timings[name],flush=True)
try:
    run('final-tests','final-tests.py')
    run('micro','bench.py')
    run('gpu-correctness','gpu-correctness.py')
    run('e2e','e2e-final.py')
    # Restore and recheck the shipped code after the final baseline block.
    run('final-tests-after-e2e','final-tests.py')
    (final/'DONE.json').write_text(json.dumps(timings,indent=2))
except BaseException as e:(final/'FAILED.json').write_text(json.dumps({'error':repr(e),'timings':timings}));raise
