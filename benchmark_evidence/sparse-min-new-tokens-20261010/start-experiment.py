import pathlib,subprocess
r=pathlib.Path('/content/min-tokens')
assert (r/'READY.json').exists()
pathlib.Path('/content/sglang-src/python/sglang/kernels/ops/sampling/min_new_tokens.py').write_bytes((r/'min_new_tokens_kernel.py').read_bytes())
p=subprocess.Popen(['/content/sglang-env/bin/python',str(r/'run-final.py')],stdout=open(r/'postguard-worker.log','w'),stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(r/'postguard-worker.pid').write_text(str(p.pid));print('EXPERIMENT_STARTED',p.pid)
