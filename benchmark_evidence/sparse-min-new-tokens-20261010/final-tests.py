import pathlib,subprocess,os,json,time
r=pathlib.Path('/content/min-tokens');target=pathlib.Path('/content/sglang-src/python/sglang/srt/sampling/penaltylib/min_new_tokens.py');target.write_bytes((r/'pr_min_new_tokens.py').read_bytes())
env={**os.environ,'PATH':'/content/sglang-env/bin:'+os.environ.get('PATH',''),'PYTHONPATH':'/content/sglang-src/python','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
t=time.monotonic()
args=['/content/sglang-env/bin/python','-m','pytest',str(r/'test_penaltylib.py'),str(r/'test_min_new_tokens.py'),'/content/sglang-src/test/registered/unit/sampling/test_sampling_batch_info.py','/content/sglang-src/test/registered/unit/sampling/test_sampling_metadata_staging.py','-q']
with open(r/'final-tests.log','w') as log:p=subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT)
(r/'final-tests.json').write_text(json.dumps({'args':args,'exit_code':p.returncode,'elapsed_s':time.monotonic()-t},indent=2))
print((r/'final-tests.log').read_text());assert p.returncode==0
